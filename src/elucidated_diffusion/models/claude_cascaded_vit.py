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
    """Fine tokens query coarse semantic tokens"""
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
# Shared-weight cascaded ViT with semantic flow
# -----------------------------
class SemanticCascadeViT(nn.Module):
    def __init__(self, dim=384, depth=3, num_heads=6, patch_size=4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        # Default unconditional semantic token
        self.default_semantic_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Time embedding (shared across all stages)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
        # Scale-specific embeddings (tells model what resolution it's processing)
        self.scale_embed_32 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_64 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_128 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_256 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Absolute positional embeddings (only at coarse scales)
        self.abs_pos_embed_32 = nn.Parameter(torch.randn(1, 64, dim) * 0.02)   # 8×8
        self.abs_pos_embed_64 = nn.Parameter(torch.randn(1, 256, dim) * 0.02)  # 16×16
        
        # Resolution-specific patch embeddings (input layer)
        self.patch_in = nn.ModuleDict({
            '32': nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            '64': nn.Conv2d(6, dim, kernel_size=patch_size, stride=patch_size),
            '128': nn.Conv2d(6, dim, kernel_size=patch_size, stride=patch_size),
            '256': nn.Conv2d(6, dim, kernel_size=patch_size, stride=patch_size),
        })
        
        # SHARED semantic cross-attention (same weights for all scales)
        self.semantic_cross_attn = SemanticCrossAttention(dim, num_heads)
        
        # SHARED transformer blocks (same weights for all scales)
        # Base depth for 32/64/128, extra blocks for 256
        self.shared_blocks = nn.ModuleList([
            TransformerBlock(
                dim, 
                num_heads, 
                window_size=8,
                shift_size=0 if i % 2 == 0 else 4
            )
            for i in range(depth)
        ])
        
        # Extra blocks for 256×256 (needs more depth for fine details)
        self.extra_blocks_256 = nn.ModuleList([
            TransformerBlock(dim, num_heads, window_size=8, shift_size=0 if i % 2 == 0 else 4)
            for i in range(2)
        ])
        
        # SHARED output normalization
        self.norm = nn.LayerNorm(dim)
        
        # Resolution-specific unpatch layers (output layer)
        self.patch_out = nn.ModuleDict({
            '32': nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size),
            '64': nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size),
            '128': nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size),
            '256': nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size),
        })
        
        # Helper dicts
        self.scale_embeds = {
            32: self.scale_embed_32,
            64: self.scale_embed_64,
            128: self.scale_embed_128,
            256: self.scale_embed_256,
        }
        
        self.abs_pos_embeds = {
            32: self.abs_pos_embed_32,
            64: self.abs_pos_embed_64,
            128: None,
            256: None,
        }
    
    def process_at_resolution(self, x, t, resolution, coarse_semantics, use_extra_blocks=False):
        """
        Process tokens at a given resolution using SHARED transformer weights
        x: input pixels
        t: timestep (already embedded)
        resolution: 32, 64, 128, or 256
        coarse_semantics: semantic tokens from previous stage
        use_extra_blocks: use extra depth for 256×256
        Returns: (pixels, semantic_tokens)
        """
        B = x.shape[0]
        res_str = str(resolution)
        tokens_per_side = resolution // self.patch_size
        
        # Patchify
        tokens = self.patch_in[res_str](x)
        h, w = tokens.shape[2], tokens.shape[3]
        tokens = rearrange('b c h w -> b (h w) c', tokens)
        
        # Add embeddings
        tokens = tokens + t[:, None, :]  # Time
        tokens = tokens + self.scale_embeds[resolution]  # Scale
        if self.abs_pos_embeds[resolution] is not None:
            tokens = tokens + self.abs_pos_embeds[resolution]  # Absolute position
        
        # Inject coarse semantics
        tokens = self.semantic_cross_attn(tokens, coarse_semantics)
        
        # SHARED transformer blocks
        for block in self.shared_blocks:
            tokens = block(tokens, h, w)
        
        # Extra blocks for 256×256
        if use_extra_blocks:
            for block in self.extra_blocks_256:
                tokens = block(tokens, h, w)
        
        # Normalize
        semantic_tokens = self.norm(tokens)
        
        # Convert to pixels
        tokens_spatial = rearrange('b (h w) c -> b c h w', semantic_tokens, h=h, w=w)
        pixels = self.patch_out[res_str](tokens_spatial)
        
        return pixels, semantic_tokens
    
    def upsample_semantics(self, semantic_tokens, from_size, to_size):
        """Upsample semantic token grid via nearest neighbor"""
        B, N, C = semantic_tokens.shape
        from_h = from_w = int(N ** 0.5)
        to_h = to_w = to_size // self.patch_size
        
        tokens_spatial = rearrange('b (h w) c -> b c h w', semantic_tokens, h=from_h, w=from_w)
        tokens_up = F.interpolate(tokens_spatial, size=(to_h, to_w), mode='nearest')
        tokens_up_flat = rearrange('b c h w -> b (h w) c', tokens_up)
        
        return tokens_up_flat
    
    def forward(self, x, t, global_conditioning=None):
        """
        x: [B, 3, H, W] - noisy input image (H, W ∈ {32, 64, 128, 256})
        t: [B] or [B, 1] - timestep
        global_conditioning: [B, M, dim] - optional semantic conditioning
        Returns: [B, 3, H, W] - denoised image
        """
        B = x.shape[0]
        target_size = x.shape[-1]
        
        # Prepare timestep embedding
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        t_emb = self.time_embed(t)
        
        # Use provided conditioning or default
        if global_conditioning is None:
            global_conditioning = self.default_semantic_token.expand(B, -1, -1)
        
        # Stage 1: 32×32
        x_32 = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        out_32, sem_32 = self.process_at_resolution(
            x_32, t_emb, 32, coarse_semantics=global_conditioning
        )
        
        if target_size == 32:
            return out_32
        
        # Stage 2: 64×64
        x_64 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        out_32_up = F.interpolate(out_32, size=(64, 64), mode='bilinear', align_corners=False)
        sem_32_up = self.upsample_semantics(sem_32, 32, 64)
        out_64, sem_64 = self.process_at_resolution(
            torch.cat([x_64, out_32_up], dim=1), t_emb, 64, coarse_semantics=sem_32_up
        )
        
        if target_size == 64:
            return out_64
        
        # Stage 3: 128×128
        x_128 = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        out_64_up = F.interpolate(out_64, size=(128, 128), mode='bilinear', align_corners=False)
        sem_64_up = self.upsample_semantics(sem_64, 64, 128)
        out_128, sem_128 = self.process_at_resolution(
            torch.cat([x_128, out_64_up], dim=1), t_emb, 128, coarse_semantics=sem_64_up
        )
        
        if target_size == 128:
            return out_128
        
        # Stage 4: 256×256 (with extra depth)
        x_256 = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        out_128_up = F.interpolate(out_128, size=(256, 256), mode='bilinear', align_corners=False)
        sem_128_up = self.upsample_semantics(sem_128, 128, 256)
        out_256, _ = self.process_at_resolution(
            torch.cat([x_256, out_128_up], dim=1), t_emb, 256, 
            coarse_semantics=sem_128_up, use_extra_blocks=True
        )
        
        return out_256