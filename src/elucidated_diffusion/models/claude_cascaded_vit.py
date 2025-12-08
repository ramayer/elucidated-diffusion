# v19 from https://claude.ai/chat/534494f4-2433-4ca2-b0bb-6e05f908c761
# good semantic understanding, but very blocky checkerboard upscaling.

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
# Relative positional bias
# -----------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )
        return relative_position_bias.permute(2, 0, 1).contiguous()


# -----------------------------
# Windowed attention
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
        self.relative_position_bias = RelativePositionBias(window_size, num_heads)
    
    def forward(self, x, h, w):
        B, N, C = x.shape
        
        # Adjust window size if grid is smaller
        ws = min(self.window_size, h, w)
        shift = min(self.shift_size, ws // 2) if self.shift_size > 0 else 0
        
        # Full attention for very small grids
        if h <= ws and w <= ws:
            qkv = self.qkv(x)
            qkv = rearrange('b n (three nh dh) -> three b nh n dh', 
                           qkv, three=3, nh=self.num_heads, dh=self.head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = dot('b nh n dh, b nh m dh -> b nh n m', q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            
            out = dot('b nh n m, b nh m dh -> b nh n dh', attn, v)
            out = rearrange('b nh n dh -> b n (nh dh)', out)
            
            return x + self.proj(out)
        
        # Windowed attention for larger grids
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
        rel_pos_bias = self.relative_position_bias()
        attn = attn + rel_pos_bias[None, None, :, :, :]
        attn = F.softmax(attn, dim=-1)
        
        out = dot('b w h n m, b w h m d -> b w h n d', attn, v)
        
        out = rearrange('b (nh nw) heads (ws1 ws2) dh -> b (nh ws1) (nw ws2) heads dh', 
                       out, nh=h//ws, nw=w//ws, ws1=ws, ws2=ws)
        out_spatial = rearrange('b h w nh dh -> b h w (nh dh)', out)
        
        if shift > 0:
            out_spatial = torch.roll(out_spatial, shifts=(shift, shift), dims=(1, 2))
        
        out_flat = rearrange('b h w c -> b (h w) c', out_spatial)
        return x + self.proj(out_flat)


# -----------------------------
# Cross-attention for semantic injection
# -----------------------------
class SemanticCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, semantic_tokens):
        B, N, C = x.shape
        
        q = self.to_q(self.norm_q(x))
        k, v = self.to_kv(self.norm_kv(semantic_tokens)).chunk(2, dim=-1)
        
        q = rearrange('b n (nh dh) -> b nh n dh', q, nh=self.num_heads, dh=self.head_dim)
        k = rearrange('b m (nh dh) -> b nh m dh', k, nh=self.num_heads, dh=self.head_dim)
        v = rearrange('b m (nh dh) -> b nh m dh', v, nh=self.num_heads, dh=self.head_dim)
        
        attn = dot('b nh n dh, b nh m dh -> b nh n m', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = dot('b nh n m, b nh m dh -> b nh n dh', attn, v)
        out = rearrange('b nh n dh -> b n (nh dh)', out)
        
        return x + self.proj(out)


# -----------------------------
# Transformer block
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
# Single resolution processor
# -----------------------------
class ResolutionProcessor(nn.Module):
    def __init__(self, in_channels, dim, depth, num_heads, patch_size, max_tokens):
        super().__init__()
        self.patch_size = patch_size
        
        # Input
        self.patch_in = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable absolute position (aspect-ratio flexible via slicing)
        self.abs_pos_embed = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        
        # Semantic cross-attention
        self.semantic_cross_attn = SemanticCrossAttention(dim, num_heads)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, window_size=8, shift_size=0 if i % 2 == 0 else 4)
            for i in range(depth)
        ])
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.patch_out = nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x, t_emb, scale_emb, coarse_semantics):
        B = x.shape[0]
        
        # Patchify
        tokens = self.patch_in(x)
        h, w = tokens.shape[2], tokens.shape[3]
        tokens = rearrange('b c h w -> b (h w) c', tokens)
        num_tokens = h * w
        
        # Add embeddings
        tokens = tokens + t_emb[:, None, :] + scale_emb
        
        # Add positional embedding (slice to actual grid size for aspect ratios)
        if num_tokens <= self.abs_pos_embed.shape[1]:
            tokens = tokens + self.abs_pos_embed[:, :num_tokens, :]
        else:
            # Interpolate if somehow we have more tokens than expected (shouldn't happen)
            pos = F.interpolate(
                self.abs_pos_embed.permute(0, 2, 1),
                size=num_tokens,
                mode='linear'
            ).permute(0, 2, 1)
            tokens = tokens + pos
        
        # Inject coarse semantics
        tokens = self.semantic_cross_attn(tokens, coarse_semantics)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, h, w)
        
        # Output
        semantic_tokens = self.norm(tokens)
        tokens_spatial = rearrange('b (h w) c -> b c h w', semantic_tokens, h=h, w=w)
        pixels = self.patch_out(tokens_spatial)
        
        return pixels, semantic_tokens


# -----------------------------
# Semantic cascade ViT (back to basics)
# -----------------------------
class SemanticCascadeViT(nn.Module):
    def __init__(self, dim=384, depth=3, num_heads=6, patch_size=4, share_weights=True):
        """
        Power-of-2 cascade: input_size / 4 → input_size / 2 → input_size
        Examples:
          - 32×32: 8×8 → 16×16 → 32×32
          - 128×128: 32×32 → 64×64 → 128×128
          - 640×64: 160×16 → 320×32 → 640×64 (aspect-ratio preserved!)
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.share_weights = share_weights
        
        # Default unconditional semantic token
        self.default_semantic_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Time embedding (always shared)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
        # Scale embeddings (coarse, medium, fine)
        self.scale_embed_coarse = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_medium = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_fine = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        if share_weights:
            # ONE processor for all resolutions (shared weights)
            # Max tokens: 128×128 / 4 patch = 32×32 = 1024 tokens
            self.processor = ResolutionProcessor(
                in_channels=6,  # Always 6 (handles first stage via zeros)
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                patch_size=patch_size,
                max_tokens=1024
            )
        else:
            # Separate processors per resolution (more capacity)
            self.processor_coarse = ResolutionProcessor(3, dim, depth, num_heads, patch_size, 1024)
            self.processor_medium = ResolutionProcessor(6, dim, depth, num_heads, patch_size, 4096)
            self.processor_fine = ResolutionProcessor(6, dim, depth+2, num_heads, patch_size, 16384)
    
    def process_resolution(self, x, t_emb, scale_emb, coarse_semantics, stage):
        """Process at one resolution"""
        if self.share_weights:
            # Pad first stage to 6 channels
            if x.shape[1] == 3:
                x = torch.cat([x, torch.zeros_like(x)], dim=1)
            return self.processor(x, t_emb, scale_emb, coarse_semantics)
        else:
            # Use appropriate processor
            if stage == 0:
                return self.processor_coarse(x, t_emb, scale_emb, coarse_semantics)
            elif stage == 1:
                return self.processor_medium(x, t_emb, scale_emb, coarse_semantics)
            else:
                return self.processor_fine(x, t_emb, scale_emb, coarse_semantics)
    
    def upsample_semantics(self, semantic_tokens, from_h, from_w, to_h, to_w):
        """Upsample semantic tokens (nearest neighbor)"""
        B, N, C = semantic_tokens.shape
        tokens_spatial = rearrange('b (h w) c -> b c h w', semantic_tokens, h=from_h, w=from_w)
        tokens_up = F.interpolate(tokens_spatial, size=(to_h, to_w), mode='nearest')
        return rearrange('b c h w -> b (h w) c', tokens_up)
    
    def forward(self, x, t, global_conditioning=None):
        """
        x: [B, 3, H, W] - input at any resolution
        t: [B] or [B, 1] - timestep
        global_conditioning: [B, M, dim] - optional conditioning
        """
        B, C, H, W = x.shape
        
        # Prepare timestep
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        t_emb = self.time_embed(t)
        
        # Use provided conditioning or default
        if global_conditioning is None:
            global_conditioning = self.default_semantic_token.expand(B, -1, -1)
        
        # Stage 0: Coarse (input_size / 4)
        h_coarse, w_coarse = H // 4, W // 4
        x_coarse = F.interpolate(x, size=(h_coarse, w_coarse), mode='bilinear', align_corners=False)
        out_coarse, sem_coarse = self.process_resolution(
            x_coarse, t_emb, self.scale_embed_coarse, global_conditioning, stage=0
        )
        
        # Stage 1: Medium (input_size / 2)
        h_medium, w_medium = H // 2, W // 2
        x_medium = F.interpolate(x, size=(h_medium, w_medium), mode='bilinear', align_corners=False)
        out_coarse_up = F.interpolate(out_coarse, size=(h_medium, w_medium), mode='bilinear', align_corners=False)
        
        tokens_h_coarse = h_coarse // self.patch_size
        tokens_w_coarse = w_coarse // self.patch_size
        tokens_h_medium = h_medium // self.patch_size
        tokens_w_medium = w_medium // self.patch_size
        sem_coarse_up = self.upsample_semantics(sem_coarse, tokens_h_coarse, tokens_w_coarse, 
                                                tokens_h_medium, tokens_w_medium)
        
        out_medium, sem_medium = self.process_resolution(
            torch.cat([x_medium, out_coarse_up], dim=1),
            t_emb, self.scale_embed_medium, sem_coarse_up, stage=1
        )
        
        # Stage 2: Fine (input_size / 1)
        out_medium_up = F.interpolate(out_medium, size=(H, W), mode='bilinear', align_corners=False)
        
        tokens_h_fine = H // self.patch_size
        tokens_w_fine = W // self.patch_size
        sem_medium_up = self.upsample_semantics(sem_medium, tokens_h_medium, tokens_w_medium,
                                                tokens_h_fine, tokens_w_fine)
        
        out_fine, _ = self.process_resolution(
            torch.cat([x, out_medium_up], dim=1),
            t_emb, self.scale_embed_fine, sem_medium_up, stage=2
        )
        
        return out_fine