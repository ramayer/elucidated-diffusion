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
    """Adaptive Layer Normalization with zero-initialized modulation"""
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(time_dim, dim * 2)
        # Zero initialization for stability
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, t_emb):
        scale, shift = self.linear(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]

class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0, use_rel_pos=True):
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
        
        # Zero init for stability (AdaLN-Zero)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
        # Relative position bias
        self.use_rel_pos = use_rel_pos
        if use_rel_pos and self.use_windowed:
            self.rel_pos_bias = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
            # Initialize relative position bias
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = coords.flatten(1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            self.register_buffer("relative_position_index", 
                               relative_coords.sum(-1))
    
    def forward(self, x, h, w):
        B, N, C = x.shape
        
        if not self.use_windowed:
            # Full attention (for coarse resolutions)
            qkv = self.qkv(x)
            qkv = rearrange('b n (three nh dh) -> three b nh n dh', 
                          qkv, three=3, nh=self.num_heads, dh=self.head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = dot('b h n d, b h m d -> b h n m', q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = dot('b h n m, b h m d -> b h n d', attn, v)
            out = rearrange('b nh n dh -> b n (nh dh)', out)
            
            return self.proj(out)
        
        # Windowed attention (for fine resolutions)
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
        
        # Add relative position bias
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

class CrossScaleAttention(nn.Module):
    """Cross-attention to previous scale's semantic tokens"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
        # Zero init for stability
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x, context):
        """
        x: [B, N, C] - current scale tokens
        context: [B, M, C] - previous scale tokens
        """
        B, N, C = x.shape
        _, M, _ = context.shape
        
        q = self.q(x)
        kv = self.kv(context)
        
        q = rearrange('b n (nh dh) -> b nh n dh', q, nh=self.num_heads, dh=self.head_dim)
        kv = rearrange('b m (two nh dh) -> two b nh m dh', kv, two=2, nh=self.num_heads, dh=self.head_dim)
        k, v = kv[0], kv[1]
        
        attn = dot('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = dot('b h n m, b h m d -> b h n d', attn, v)
        out = rearrange('b nh n dh -> b n (nh dh)', out)
        
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, time_dim, num_heads=8, window_size=8, shift_size=0, 
                 mlp_ratio=4, use_rel_pos=True, use_cross_attn=False):
        super().__init__()
        self.adaln1 = AdaLN(dim, time_dim)
        self.attn = WindowedAttention(dim, num_heads, window_size, shift_size, use_rel_pos)
        
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.adaln_cross = AdaLN(dim, time_dim)
            self.cross_attn = CrossScaleAttention(dim, num_heads)
        
        self.adaln2 = AdaLN(dim, time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        # Zero init MLP output
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x, h, w, t_emb, context=None):
        # Self-attention with AdaLN
        x = x + self.attn(self.adaln1(x, t_emb), h, w)
        
        # Cross-attention to previous scale (if available)
        if self.use_cross_attn and context is not None:
            x = x + self.cross_attn(self.adaln_cross(x, t_emb), context)
        
        # MLP with AdaLN
        x = x + self.mlp(self.adaln2(x, t_emb))
        
        return x

class CNNEncoder(nn.Module):
    """CNN encoder to process pixels before patchification"""
    def __init__(self, in_channels=3, dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, 3, padding=1),
            nn.GroupNorm(8, dim // 2),
            nn.SiLU(),
            nn.Conv2d(dim // 2, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class CNNDecoder(nn.Module):
    """CNN decoder to generate pixels from tokens"""
    def __init__(self, dim=512, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GroupNorm(8, dim // 2),
            nn.SiLU(),
            nn.Conv2d(dim // 2, out_channels, 1)
        )
    
    def forward(self, x):
        return self.conv(x)

class MultiScaleSemanticViT(nn.Module):
    def __init__(self, dim=512, depth=6, num_heads=8, patch_size=2):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4)
        )
        time_dim = dim * 4
        
        # Scale embeddings (learnable)
        self.scale_embed_64 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_128 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale_embed_256 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Positional embeddings (absolute for 64x64, relative bias for others)
        self.pos_embed_64 = nn.Parameter(torch.randn(1, 1024, dim) * 0.02)  # 32×32 tokens
        self.pos_embed_128 = nn.Parameter(torch.randn(1, 4096, dim) * 0.02)  # 64×64 tokens
        self.pos_embed_256 = nn.Parameter(torch.randn(1, 16384, dim) * 0.02)  # 128×128 tokens
        
        # CNN encoders per scale
        self.cnn_encoder_64 = CNNEncoder(3, dim)
        self.cnn_encoder_128 = CNNEncoder(3, dim)
        self.cnn_encoder_256 = CNNEncoder(3, dim)
        
        # Patchification
        self.patch_embed = nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size)
        
        # Transformer blocks for 64×64 (full attention, no cross-attention)
        self.blocks_64 = nn.ModuleList([
            TransformerBlock(
                dim, time_dim,
                num_heads=num_heads,
                window_size=0,  # Full attention
                shift_size=0,
                use_rel_pos=False,  # Use absolute pos embeddings
                use_cross_attn=False
            )
            for _ in range(depth)
        ])
        
        # Transformer blocks for 128×128 (windowed + cross-attention)
        self.blocks_128 = nn.ModuleList([
            TransformerBlock(
                dim, time_dim,
                num_heads=num_heads,
                window_size=8,
                shift_size=0 if i % 2 == 0 else 4,
                use_rel_pos=True,
                use_cross_attn=True
            )
            for i in range(depth)
        ])
        
        # Transformer blocks for 256×256 (windowed + cross-attention)
        self.blocks_256 = nn.ModuleList([
            TransformerBlock(
                dim, time_dim,
                num_heads=num_heads,
                window_size=8,
                shift_size=0 if i % 2 == 0 else 4,
                use_rel_pos=True,
                use_cross_attn=True
            )
            for i in range(depth)
        ])
        
        # CNN decoders per scale
        self.decoder_64 = CNNDecoder(dim, 3)
        self.decoder_128 = CNNDecoder(dim, 3)
        self.decoder_256 = CNNDecoder(dim, 3)
        
        self.scale_configs = {
            64: {
                'encoder': self.cnn_encoder_64,
                'blocks': self.blocks_64,
                'decoder': self.decoder_64,
                'scale_embed': self.scale_embed_64,
                'pos_embed': self.pos_embed_64
            },
            128: {
                'encoder': self.cnn_encoder_128,
                'blocks': self.blocks_128,
                'decoder': self.decoder_128,
                'scale_embed': self.scale_embed_128,
                'pos_embed': self.pos_embed_128
            },
            256: {
                'encoder': self.cnn_encoder_256,
                'blocks': self.blocks_256,
                'decoder': self.decoder_256,
                'scale_embed': self.scale_embed_256,
                'pos_embed': self.pos_embed_256
            }
        }
    
    def forward_single_scale(self, x, t_emb, resolution, prev_tokens=None):
        """
        Process a single scale, returning semantic tokens and pixel output
        
        Args:
            x: [B, 3, H, W] input at this scale
            t_emb: [B, time_dim] time embedding
            resolution: target resolution (64, 128, or 256)
            prev_tokens: [B, M, dim] semantic tokens from previous scale
        
        Returns:
            tokens: [B, N, dim] semantic representation
            pixels: [B, 3, H, W] pixel output
        """
        config = self.scale_configs[resolution]
        
        # CNN encoder
        features = config['encoder'](x)  # [B, dim, H, W]
        
        # Patchify
        tokens = self.patch_embed(features)  # [B, dim, h, w]
        h, w = tokens.shape[2], tokens.shape[3]
        tokens = rearrange('b c h w -> b (h w) c', tokens)
        
        # Add embeddings
        tokens = tokens + config['scale_embed'] + config['pos_embed']
        
        # Transformer blocks
        for block in config['blocks']:
            tokens = block(tokens, h, w, t_emb, context=prev_tokens)
        
        # Unpatchify
        tokens_spatial = rearrange('b (h w) c -> b c h w', tokens, h=h, w=w)
        
        # Upsample to original resolution
        tokens_upsampled = F.interpolate(tokens_spatial, size=(resolution, resolution), 
                                        mode='bilinear', align_corners=False)
        
        # CNN decoder to pixels
        pixels = config['decoder'](tokens_upsampled)
        
        return tokens, pixels
    
    def forward(self, x, t):
        """
        Cascaded multi-scale processing with semantic token passing
        
        Args:
            x: [B, 3, H, W] input image (H=W must be 64, 128, or 256)
            t: [B] timesteps
        
        Returns:
            If training: dict with outputs at each scale
            If inference: final pixel output at target resolution
        """
        target_size = x.shape[-1]
        assert target_size in [64, 128, 256], "Input must be 64x64, 128x128, or 256x256"
        
        # Time embedding
        t_emb = self.time_embed(t)  # [B, time_dim]
        
        # Scale 64 (coarsest - full attention for global understanding)
        x_64 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        tokens_64, pixels_64 = self.forward_single_scale(x_64, t_emb, 64, prev_tokens=None)
        
        if target_size == 64:
            return pixels_64
            return {'64': pixels_64} if self.training else pixels_64
        
        # Scale 128 (cross-attend to tokens_64)
        x_128 = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        tokens_128, pixels_128 = self.forward_single_scale(x_128, t_emb, 128, prev_tokens=tokens_64)
        
        if target_size == 128:
            return pixels_128
            return {'64': pixels_64, '128': pixels_128} if self.training else pixels_128
        
        # Scale 256 (cross-attend to tokens_128)
        x_256 = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        tokens_256, pixels_256 = self.forward_single_scale(x_256, t_emb, 256, prev_tokens=tokens_128)
        
        if self.training:
            return pixels_256
            return {'64': pixels_64, '128': pixels_128, '256': pixels_256}
        else:
            return pixels_256