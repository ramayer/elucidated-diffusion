# https://claude.ai/share/fb8d088e-9d82-4715-b7c1-44bd727e0278
# TODO - try this again - it showed some promise but was noisy.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einx import rearrange, dot

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freq = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half_dim) / (half_dim - 1))
        self.register_buffer('freq', freq)

    def forward(self, x):
        x = x[:, None] * self.freq[None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = self.head_dim ** -0.5
        self.use_windowed = window_size > 0
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
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
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0, mlp_ratio=4, resid_scale=0.5):
        super().__init__()
        self.resid_scale = resid_scale
        
        self.norm1 = RMSNorm(dim)
        self.attn = WindowedAttention(dim, num_heads, window_size, shift_size)
        
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
        self.adaln_zero_attn = nn.Parameter(torch.zeros(1))
        self.adaln_zero_mlp = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, h, w):
        x = x + self.resid_scale * self.adaln_zero_attn * self.attn(self.norm1(x), h, w)
        x = x + self.resid_scale * self.adaln_zero_mlp * self.mlp(self.norm2(x))
        return x

class PatchDiffusion(nn.Module):
    def __init__(self, resolution, patch_size=4, dim=384, depth=6, num_heads=8, resid_scale=0.5):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.dim = dim
        self.tokens_per_side = resolution // patch_size
        
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        nn.init.kaiming_normal_(self.patch_embed.weight, nonlinearity='relu')
        nn.init.zeros_(self.patch_embed.bias)
        
        self.semantic_proj = nn.Linear(dim * 2, dim)
        nn.init.trunc_normal_(self.semantic_proj.weight, std=0.02)
        nn.init.zeros_(self.semantic_proj.bias)
        
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
                resid_scale=resid_scale
            )
            for i in range(depth)
        ])
        
        self.norm = RMSNorm(dim)
        
        self.unpatch = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='nearest'),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim // 2, 1),
            nn.GroupNorm(8, dim // 2),
            nn.SiLU(),
            nn.Conv2d(dim // 2, 3, 3, padding=1)
        )
        
        for m in self.unpatch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
        pixels = self.unpatch(tokens_spatial)
        
        return pixels, tokens

class ResearchBackedHybridViT(nn.Module):
    def __init__(self, dim=384, depth=6, num_heads=8, patch_size=4, resid_scale=0.5):
        super().__init__()
        self.patch_size = patch_size
        
        self.model_32 = PatchDiffusion(32, patch_size, dim, depth, num_heads, resid_scale)
        self.model_64 = PatchDiffusion(64, patch_size, dim, depth, num_heads, resid_scale)
        self.model_128 = PatchDiffusion(128, patch_size, dim, depth, num_heads, resid_scale)
        self.model_256 = PatchDiffusion(256, patch_size, dim, depth + 2, num_heads, resid_scale)
    
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
    