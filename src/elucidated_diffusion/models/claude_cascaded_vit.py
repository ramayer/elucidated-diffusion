# v5

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
# Relative positional bias (Swin-style)
# -----------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        
        # Relative position table: (2*window_size-1) × (2*window_size-1)
        # For window_size=8: 15×15 = 225 possible relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Pre-compute relative position indices
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
        
        # Relative coordinates: [2, ws*ws, ws*ws]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
        
        # Shift to start from 0
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        
        relative_position_index = relative_coords.sum(-1)  # [ws*ws, ws*ws]
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Initialize bias with small values
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self):
        # Get bias for this window: [ws*ws, ws*ws, num_heads]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )
        return relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, ws*ws, ws*ws]


# -----------------------------
# Windowed multi-head attention with relative position bias
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
        
        # Relative position bias
        self.relative_position_bias = RelativePositionBias(window_size, num_heads)
    
    def forward(self, x, h, w):
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
        
        # Attention within windows with relative position bias
        attn = dot('b w h n d, b w h m d -> b w h n m', q, k) * self.scale
        
        # Add relative position bias: [num_heads, ws*ws, ws*ws]
        rel_pos_bias = self.relative_position_bias()  # [nh, ws*ws, ws*ws]
        attn = attn + rel_pos_bias[None, None, :, :, :]  # Broadcast to [B, num_windows, nh, ws*ws, ws*ws]
        
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
# Transformer block with windowed attention
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
# Multi-scale shared transformer with hybrid positional encoding
# -----------------------------
class MultiScaleSharedViT(nn.Module):
    def __init__(self, dim=512, depth=6, num_heads=8, patch_size=4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
        # Scale-specific embeddings (learnable)
        self.scale_embed_32 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_64 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_128 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_256 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Absolute positional embeddings ONLY at coarse scales for global layout
        # Finer scales rely on: (1) relative position bias, (2) cascade from coarse scales
        self.abs_pos_embed_32 = nn.Parameter(torch.randn(1, 64, dim) * 0.02)    # 8×8 tokens - ESSENTIAL
        self.abs_pos_embed_64 = nn.Parameter(torch.randn(1, 256, dim) * 0.02)   # 16×16 tokens - helpful
        # No absolute position for 128×128 and 256×256 - they refine coarse structure
        
        # Shared transformer blocks with relative position bias
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim, 
                num_heads=num_heads, 
                window_size=8,
                shift_size=0 if i % 2 == 0 else 4
            )
            for i in range(depth)
        ])
        
        # Resolution-specific input/output projections
        self.patch_in = nn.ModuleDict({
            '32': nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            '64': nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            '128': nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            '256': nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        })
        
        self.norm = nn.LayerNorm(dim)
        
        self.patch_out = nn.ModuleDict({
            '32': nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size),
            '64': nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size),
            '128': nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size),
            '256': nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size),
        })
        
        # Map for embeddings
        self.scale_embeds = {
            32: self.scale_embed_32,
            64: self.scale_embed_64,
            128: self.scale_embed_128,
            256: self.scale_embed_256,
        }
        
        self.abs_pos_embeds = {
            32: self.abs_pos_embed_32,
            64: self.abs_pos_embed_64,
            128: None,  # No absolute position - relies on cascade
            256: None,  # No absolute position - pure detail refinement
        }
    
    def forward_single_scale(self, x, t, resolution):
        """Process at a single resolution using shared transformer"""
        B = x.shape[0]
        res_str = str(resolution)
        
        # Time embedding
        t_emb = self.time_embed(t)  # [B, dim]
        
        # Patchify
        tokens = self.patch_in[res_str](x)  # [B, dim, h, w]
        h, w = tokens.shape[2], tokens.shape[3]
        tokens = rearrange('b c h w -> b (h w) c', tokens)
        
        # Add embeddings:
        # 1. Time embedding (what timestep are we at?)
        # 2. Scale embedding (what resolution level are we at?)
        # 3. Absolute position embedding (where in the image?) - only at coarse scales
        abs_pos = self.abs_pos_embeds[resolution]
        if abs_pos is not None:
            tokens = tokens + t_emb[:, None, :] + self.scale_embeds[resolution] + abs_pos
        else:
            tokens = tokens + t_emb[:, None, :] + self.scale_embeds[resolution]
        
        # Shared transformer blocks (with relative position bias inside)
        for block in self.blocks:
            tokens = block(tokens, h, w)
        
        # Normalize and unpatchify
        tokens = self.norm(tokens)
        tokens = rearrange('b (h w) c -> b c h w', tokens, h=h, w=w)
        out = self.patch_out[res_str](tokens)
        
        return out
    
    def forward(self, x, t):
        """Cascaded multi-scale processing: 32 → 64 → 128 → 256"""
        target_size = x.shape[-1]
        
        # Stage 1: Generate base at 32x32
        x_32 = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        out_32 = self.forward_single_scale(x_32, t, resolution=32)
        
        if target_size == 32:
            return out_32
        
        # Stage 2: Refine at 64x64
        x_64 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        out_32_up = F.interpolate(out_32, size=(64, 64), mode='bilinear', align_corners=False)
        out_64 = self.forward_single_scale(x_64 + out_32_up, t, resolution=64)
        
        if target_size == 64:
            return out_64
        
        # Stage 3: Add details at 128x128
        x_128 = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        out_64_up = F.interpolate(out_64, size=(128, 128), mode='bilinear', align_corners=False)
        out_128 = self.forward_single_scale(x_128 + out_64_up, t, resolution=128)
        
        if target_size == 128:
            return out_128
        
        # Stage 4: Final details at 256x256
        x_256 = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        out_128_up = F.interpolate(out_128, size=(256, 256), mode='bilinear', align_corners=False)
        out_256 = self.forward_single_scale(x_256 + out_128_up, t, resolution=256)
        
        return out_256


# -----------------------------
# Convenience wrapper
# -----------------------------
class UNet128(nn.Module):
    """Drop-in replacement for training code that expects UNet128"""
    def __init__(self, in_channels=3, base_ch=64, emb_dim=128):
        super().__init__()
        # Map base_ch to embedding dim: base_ch=64 -> dim=512
        dim = base_ch * 8
        self.model = MultiScaleSharedViT(dim=dim, depth=6, num_heads=8)
    
    def forward(self, x, t):
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        return self.model(x, t)