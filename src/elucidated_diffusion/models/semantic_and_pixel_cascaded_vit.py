import torch
import torch.nn as nn
import torch.nn.functional as F
from einx import rearrange, dot

# Slow to train - feels like it's headed an OK direction, but 
# otehr models seem to converge much faster?

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0, use_rel_pos=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = self.head_dim ** -0.5
        self.use_windowed = window_size > 0
        self.use_rel_pos = use_rel_pos
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        if use_rel_pos and self.use_windowed:
            self.rel_pos_bias = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = coords.flatten(1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            self.register_buffer("relative_position_index", relative_coords.sum(-1))
    
    def forward(self, x, h, w):
        B, N, C = x.shape
        
        if not self.use_windowed:
            qkv = self.qkv(x)
            qkv = rearrange('b n (three nh dh) -> three b nh n dh', 
                          qkv, three=3, nh=self.num_heads, dh=self.head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = dot('b h n d, b h m d -> b h n m', q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = dot('b h n m, b h m d -> b h n d', attn, v)
            out = rearrange('b nh n dh -> b n (nh dh)', out)
            
            return self.proj(out)
        
        ws = self.window_size
        shift = self.shift_size
        
        x_spatial = rearrange('b (h w) c -> b h w c', x, h=h, w=w)
        
        if shift > 0:
            x_spatial = torch.roll(x_spatial, shifts=(-shift, -shift), dims=(1, 2))
        
        x_windows = rearrange('b h w c -> b (h w) c', x_spatial)
        
        qkv = self.qkv(x_windows)
        qkv = rearrange('b (h w) (three nh dh) -> three b (h w) nh dh', 
                       qkv, three=3, nh=self.num_heads, dh=self.head_dim, h=h, w=w)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = rearrange('b (h w) nh dh -> b h w nh dh', q, h=h, w=w)
        q = rearrange('b (nh ws1) (nw ws2) heads dh -> b (nh nw) heads (ws1 ws2) dh', 
                     q, ws1=ws, ws2=ws)
        
        k = rearrange('b (h w) nh dh -> b h w nh dh', k, h=h, w=w)
        k = rearrange('b (nh ws1) (nw ws2) heads dh -> b (nh nw) heads (ws1 ws2) dh', 
                     k, ws1=ws, ws2=ws)
        
        v = rearrange('b (h w) nh dh -> b h w nh dh', v, h=h, w=w)
        v = rearrange('b (nh ws1) (nw ws2) heads dh -> b (nh nw) heads (ws1 ws2) dh', 
                     v, ws1=ws, ws2=ws)
        
        attn = dot('b w h n d, b w h m d -> b w h n m', q, k) * self.scale
        
        if self.use_rel_pos:
            rel_pos_bias = self.rel_pos_bias[self.relative_position_index.view(-1)].view(
                ws * ws, ws * ws, -1)
            rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
            attn = attn + rel_pos_bias.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        
        out = dot('b w h n m, b w h m d -> b w h n d', attn, v)
        
        out = rearrange('b (nh nw) heads (ws1 ws2) dh -> b (nh ws1) (nw ws2) heads dh', 
                       out, nh=h//ws, nw=w//ws, ws1=ws, ws2=ws)
        out_spatial = rearrange('b h w nh dh -> b h w (nh dh)', out)
        
        if shift > 0:
            out_spatial = torch.roll(out_spatial, shifts=(shift, shift), dims=(1, 2))
        
        out_flat = rearrange('b h w c -> b (h w) c', out_spatial)
        return self.proj(out_flat)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0, mlp_ratio=4, use_rel_pos=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(dim, num_heads, window_size, shift_size, use_rel_pos)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    
    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchDiffusion(nn.Module):
    def __init__(self, resolution, patch_size=4, dim=384, depth=6, num_heads=8):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.dim = dim
        self.tokens_per_side = resolution // patch_size
        
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        self.semantic_proj = nn.Linear(dim * 2, dim)
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
        tokens_per_side = resolution // patch_size
        window_size = min(8, tokens_per_side)
        use_windowed = tokens_per_side > 8
        
        if not use_windowed:
            num_tokens = tokens_per_side ** 2
            self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        else:
            self.pos_embed = None
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim, num_heads,
                window_size=window_size if use_windowed else 0,
                shift_size=0 if not use_windowed or i % 2 == 0 else window_size // 2,
                mlp_ratio=4,
                use_rel_pos=use_windowed
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
        self.token_blend = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU()
        )
        
        self.unpatch = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='nearest'),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GroupNorm(8, dim // 2),
            nn.SiLU(),
            nn.Conv2d(dim // 2, 3, 3, padding=1)
        )
    
    def forward(self, x_pixels, t, semantic_context=None):
        B = x_pixels.shape[0]
        
        t_emb = self.time_embed(t)
        
        tokens = self.patch_embed(x_pixels)
        h, w = tokens.shape[2], tokens.shape[3]
        tokens = rearrange('b c h w -> b (h w) c', tokens)
        
        if semantic_context is not None:
            sem_h = sem_w = int(semantic_context.shape[1] ** 0.5)
            semantic_spatial = rearrange('b (h w) c -> b c h w', semantic_context, h=sem_h, w=sem_w)
            semantic_up = F.interpolate(semantic_spatial, size=(h, w), mode='bilinear', align_corners=False)
            semantic_up = rearrange('b c h w -> b (h w) c', semantic_up)
            
            tokens = torch.cat([tokens, semantic_up], dim=-1)
            tokens = self.semantic_proj(tokens)
        
        tokens = tokens + t_emb[:, None, :]
        
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        
        for block in self.blocks:
            tokens = block(tokens, h, w)
        
        tokens = self.norm(tokens)
        
        tokens_spatial = rearrange('b (h w) c -> b c h w', tokens, h=h, w=w)
        tokens_spatial = tokens_spatial + self.token_blend(tokens_spatial)
        
        pixels = self.unpatch(tokens_spatial)
        
        return pixels, tokens

class HybridCascadedViT(nn.Module):
    def __init__(self, dim=384, depth=6, num_heads=8, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        
        self.model_32 = PatchDiffusion(32, patch_size, dim, depth, num_heads)
        self.model_64 = PatchDiffusion(64, patch_size, dim, depth, num_heads)
        self.model_128 = PatchDiffusion(128, patch_size, dim, depth, num_heads)
        self.model_256 = PatchDiffusion(256, patch_size, dim, depth + 2, num_heads)
    
    def forward(self, x, t):
        target_size = x.shape[-1]
        
        x_32 = F.interpolate(x, size=(32, 32), mode='nearest')
        out_32, sem_32 = self.model_32(x_32, t, semantic_context=None)
        
        if target_size == 32:
            return out_32
        
        x_64 = F.interpolate(x, size=(64, 64), mode='nearest')
        out_32_up = F.interpolate(out_32, size=(64, 64), mode='bilinear', align_corners=False)
        out_64, sem_64 = self.model_64(x_64 + out_32_up, t, semantic_context=sem_32)
        
        if target_size == 64:
            return out_64
        
        x_128 = F.interpolate(x, size=(128, 128), mode='nearest')
        out_64_up = F.interpolate(out_64, size=(128, 128), mode='bilinear', align_corners=False)
        out_128, sem_128 = self.model_128(x_128 + out_64_up, t, semantic_context=sem_64)
        
        if target_size == 128:
            return out_128
        
        x_256 = F.interpolate(x, size=(256, 256), mode='nearest')
        out_128_up = F.interpolate(out_128, size=(256, 256), mode='bilinear', align_corners=False)
        out_256, sem_256 = self.model_256(x_256 + out_128_up, t, semantic_context=sem_128)
        
        return out_256