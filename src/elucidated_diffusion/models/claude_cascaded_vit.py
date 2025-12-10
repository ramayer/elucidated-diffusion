import torch
import torch.nn as nn
import torch.nn.functional as F
from einx import rearrange, dot

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

class AdaLN(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(time_dim, dim * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, t_emb):
        scale, shift = self.linear(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]

class FullAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange('b n (three nh dh) -> three b nh n dh', 
                      qkv, three=3, nh=self.num_heads, dh=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = dot('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = dot('b h n m, b h m d -> b h n d', attn, v)
        out = rearrange('b nh n dh -> b n (nh dh)', out)
        
        return self.proj(out)

class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
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

class CoarseBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.adaln1 = AdaLN(dim, dim)
        self.attn = FullAttention(dim, num_heads)
        self.adaln2 = AdaLN(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x, t_emb):
        x = x + self.attn(self.adaln1(x, t_emb))
        x = x + self.mlp(self.adaln2(x, t_emb))
        return x

class FineBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0, mlp_ratio=4):
        super().__init__()
        self.adaln1 = AdaLN(dim, dim)
        self.attn = WindowedAttention(dim, num_heads, window_size, shift_size)
        self.adaln2 = AdaLN(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x, h, w, t_emb):
        x = x + self.attn(self.adaln1(x, t_emb), h, w)
        x = x + self.mlp(self.adaln2(x, t_emb))
        return x

class CoarseModel(nn.Module):
    def __init__(self, dim=384, depth=4, num_heads=6, patch_size=8):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        self.patch_in = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        tokens_per_side = 64 // patch_size
        num_tokens = tokens_per_side ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            CoarseBlock(dim, num_heads)
            for _ in range(depth)
        ])
    
    def forward(self, x_64, t_emb):
        tokens = self.patch_in(x_64)
        tokens = rearrange('b c h w -> b (h w) c', tokens)
        tokens = tokens + self.pos_embed
        
        for block in self.blocks:
            tokens = block(tokens, t_emb)
        
        return tokens

class FineModel(nn.Module):
    def __init__(self, dim=512, depth=6, num_heads=8, patch_size=8, target_size=128):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.target_size = target_size
        
        self.patch_in = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.context_proj = nn.Linear(512, dim)
        
        tokens_per_side = target_size // patch_size
        num_tokens = tokens_per_side ** 2
        coarse_tokens_per_side = 64 // patch_size
        scale_factor = tokens_per_side // coarse_tokens_per_side
        
        self.rel_pos_embed = nn.Parameter(
            torch.randn(1, scale_factor * scale_factor, dim) * 0.02
        )
        
        self.blocks = nn.ModuleList([
            FineBlock(dim, num_heads, window_size=8, 
                     shift_size=0 if i % 2 == 0 else 4)
            for i in range(depth)
        ])
        
        self.semantic_blend = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(8, dim // 2),
            nn.SiLU(),
            nn.Conv2d(dim // 2, 3, 1)
        )
    
    def forward(self, x, t_emb, coarse_tokens):
        B, C, H, W = x.shape
        tokens = self.patch_in(x)
        h, w = tokens.shape[2], tokens.shape[3]
        tokens = rearrange('b c h w -> b (h w) c', tokens)
        
        coarse_h = coarse_w = 64 // self.patch_size
        coarse_spatial = rearrange('b (h w) c -> b c h w', coarse_tokens, h=coarse_h, w=coarse_w)
        coarse_upsampled = F.interpolate(coarse_spatial, size=(h, w), mode='bilinear', align_corners=False)
        coarse_upsampled = rearrange('b c h w -> b (h w) c', coarse_upsampled)
        coarse_upsampled = self.context_proj(coarse_upsampled)
        
        scale_factor = h // coarse_h
        rel_pos = self.rel_pos_embed.repeat(B, (h // scale_factor) * (w // scale_factor), 1)
        
        tokens = tokens + coarse_upsampled + rel_pos
        
        for block in self.blocks:
            tokens = block(tokens, h, w, t_emb)
        
        tokens_spatial = rearrange('b (h w) c -> b c h w', tokens, h=h, w=w)
        tokens_spatial = tokens_spatial + self.semantic_blend(tokens_spatial)
        
        pixels = self.decoder(tokens_spatial)
        
        return pixels

class CoarseToFineViT(nn.Module):
    def __init__(self, coarse_dim=384, coarse_depth=4, fine_dim=512, fine_depth=6, 
                 num_heads=8, patch_size=8, target_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.target_size = target_size
        
        self.time_embed_coarse = nn.Sequential(
            SinusoidalPosEmb(coarse_dim),
            nn.Linear(coarse_dim, coarse_dim),
            nn.GELU()
        )
        
        self.time_embed_fine = nn.Sequential(
            SinusoidalPosEmb(fine_dim),
            nn.Linear(fine_dim, fine_dim),
            nn.GELU()
        )
        
        self.coarse = CoarseModel(coarse_dim, coarse_depth, num_heads, patch_size)
        self.fine = FineModel(fine_dim, fine_depth, num_heads, patch_size, target_size)
    
    def forward(self, x, t):
        B, C, H, W = x.shape
        assert H % 64 == 0 and W % 64 == 0, "Height and width must be multiples of 64"
        
        t_emb_coarse = self.time_embed_coarse(t)
        t_emb_fine = self.time_embed_fine(t)
        
        x_64 = F.interpolate(x, size=(64, 64), mode='nearest')
        coarse_tokens = self.coarse(x_64, t_emb_coarse)
        
        pixels = self.fine(x, t_emb_fine, coarse_tokens)
        
        if pixels.shape[-2:] != (H, W):
            pixels = F.interpolate(pixels, size=(H, W), mode='bilinear', align_corners=False)
        
        return pixels