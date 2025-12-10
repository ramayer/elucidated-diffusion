import torch
import torch.nn as nn
import torch.nn.functional as F
from einx import rearrange, dot

# -----------------------------
# Positional embedding for timestep
# -----------------------------
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


# -----------------------------
# Windowed multi-head attention with optional shifting (Swin style)
# -----------------------------
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
    
    def forward(self, x, h, w):
        # x: [B, H*W, C]
        B, N, C = x.shape
        ws = self.window_size
        shift = self.shift_size
        
        # Reshape to spatial
        x_spatial = rearrange('b (h w) c -> b h w c', x, h=h, w=w)
        
        # Cyclic shift if needed
        if shift > 0:
            x_spatial = torch.roll(x_spatial, shifts=(-shift, -shift), dims=(1, 2))
        
        x_windows = rearrange('b h w c -> b (h w) c', x_spatial)
        
        # Generate Q, K, V and split heads
        qkv = self.qkv(x_windows)
        qkv = rearrange('b (h w) (three nh dh) -> three b (h w) nh dh', 
                       qkv, three=3, nh=self.num_heads, dh=self.head_dim, h=h, w=w)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Reshape into windows
        q = rearrange('b (h w) nh dh -> b h w nh dh', q, h=h, w=w)
        q = rearrange('b (nh ws1) (nw ws2) heads dh -> b (nh nw) heads (ws1 ws2) dh', 
                     q, ws1=ws, ws2=ws)
        
        k = rearrange('b (h w) nh dh -> b h w nh dh', k, h=h, w=w)
        k = rearrange('b (nh ws1) (nw ws2) heads dh -> b (nh nw) heads (ws1 ws2) dh', 
                     k, ws1=ws, ws2=ws)
        
        v = rearrange('b (h w) nh dh -> b h w nh dh', v, h=h, w=w)
        v = rearrange('b (nh ws1) (nw ws2) heads dh -> b (nh nw) heads (ws1 ws2) dh', 
                     v, ws1=ws, ws2=ws)
        
        # Attention within windows
        attn = dot('b w h n d, b w h m d -> b w h n m', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = dot('b w h n m, b w h m d -> b w h n d', attn, v)
        
        # Reshape back
        out = rearrange('b (nh nw) heads (ws1 ws2) dh -> b (nh ws1) (nw ws2) heads dh', 
                       out, nh=h//ws, nw=w//ws, ws1=ws, ws2=ws)
        out_spatial = rearrange('b h w nh dh -> b h w (nh dh)', out)
        
        # Reverse cyclic shift
        if shift > 0:
            out_spatial = torch.roll(out_spatial, shifts=(shift, shift), dims=(1, 2))
        
        out_flat = rearrange('b h w c -> b (h w) c', out_spatial)
        return x + self.proj(out_flat)


# -----------------------------
# Transformer block with windowed attention and optional shifting
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(dim, num_heads, window_size, shift_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    
    def forward(self, x, h, w):
        x = self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# Single-resolution ViT diffusion model
# -----------------------------
class PatchDiffusion(nn.Module):
    def __init__(self, resolution, patch_size=4, dim=256, depth=4, num_heads=8):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_patches = (resolution // patch_size) ** 2
        self.tokens_per_side = resolution // patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
        # Transformer blocks with alternating regular/shifted windows (Swin style)
        window_size = min(8, self.tokens_per_side)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim, 
                num_heads, 
                window_size, 
                shift_size=0 if i % 2 == 0 else window_size // 2
            )
            for i in range(depth)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(dim)
        self.unpatch = nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x, t):
        B = x.shape[0]
        
        # Time embedding
        t_emb = self.time_embed(t)  # [B, dim]
        
        # Patchify
        x = self.patch_embed(x)  # [B, dim, h, w]
        h, w = x.shape[2], x.shape[3]
        x = rearrange('b c h w -> b (h w) c', x)
        
        # Add time embedding to each token
        x = x + t_emb[:, None, :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, h, w)
        
        # Normalize and unpatchify
        x = self.norm(x)
        x = rearrange('b (h w) c -> b c h w', x, h=h, w=w)
        x = self.unpatch(x)
        
        return x


# -----------------------------
# Cascaded ViT: 32 → 64 → 128 → 256
# -----------------------------
class CascadedViT(nn.Module):
    def __init__(self, dim=256, depth=4):
        super().__init__()
        
        # Four stages with progressively more tokens
        self.model_32 = PatchDiffusion(32, patch_size=4, dim=dim, depth=depth)      # 8x8 tokens
        self.model_64 = PatchDiffusion(64, patch_size=4, dim=dim, depth=depth)      # 16x16 tokens
        self.model_128 = PatchDiffusion(128, patch_size=4, dim=dim, depth=depth)    # 32x32 tokens
        self.model_256 = PatchDiffusion(256, patch_size=4, dim=dim, depth=depth+2)  # 64x64 tokens, deeper for details
    
    def forward(self, x, t):
        target_size = x.shape[-1]
        
        # Stage 1: Generate base at 32x32
        x_32 = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        out_32 = self.model_32(x_32, t)
        
        if target_size == 32:
            return out_32
        
        # Stage 2: Refine at 64x64
        x_64 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        out_32_up = F.interpolate(out_32, size=(64, 64), mode='bilinear', align_corners=False)
        out_64 = self.model_64(x_64 + out_32_up, t)
        
        if target_size == 64:
            return out_64
        
        # Stage 3: Add details at 128x128
        x_128 = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        out_64_up = F.interpolate(out_64, size=(128, 128), mode='bilinear', align_corners=False)
        out_128 = self.model_128(x_128 + out_64_up, t)
        
        if target_size == 128:
            return out_128
        
        # Stage 4: Final details at 256x256
        x_256 = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        out_128_up = F.interpolate(out_128, size=(256, 256), mode='bilinear', align_corners=False)
        out_256 = self.model_256(x_256 + out_128_up, t)
        
        return out_256


# -----------------------------
# Training-friendly wrapper that can work at any resolution
# -----------------------------
class FlexibleCascadedViT(nn.Module):
    """Can train and generate at 32, 64, 128, or 256"""
    def __init__(self, dim=256, depth=4):
        super().__init__()
        self.cascade = CascadedViT(dim, depth)
    
    def forward(self, x, t):
        return self.cascade(x, t)