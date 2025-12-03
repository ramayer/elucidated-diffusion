import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Note: Install with: pip install einx
try:
    import einx
    from einx import rearrange, dot, add
except ImportError:
    raise ImportError("Please install einx: pip install einx")

"""
Adaptive Semantic Coordinator UNet - Biologically Inspired, Fully Parameterized

This architecture uses loops to construct encoder/decoder of arbitrary depth,
allowing exploration of pure CNN, pure ViT, or hybrid architectures while
maintaining the biological visual cortex inspiration.

Biological Vision Hierarchy (Flexible Mapping):
================================================

V1/V2 (Primary/Secondary Visual Cortex):
  - Local processing: edges, textures, colors
  - Small receptive fields
  → Controlled by: cnn_layers parameter
  → More layers = deeper V1/V2-like processing before abstraction

V4 (Visual Area 4):
  - Object parts, intermediate features
  → Transition zone in deeper CNN layers

IT (Inferotemporal Cortex):
  - Abstract semantic understanding
  - Large receptive fields, view-invariant
  - "This is a standing cat" not "edge at position X"
  → Controlled by: vit_layers parameter
  → More layers = deeper semantic reasoning

Feedback Connections:
  - IT sends predictions back to V1/V2
  → Semantic injection to all decoder levels

Architectural Flexibility:
==========================
- Pure CNN (vit_layers=0): Fast, local-only, like early mammals
- Pure ViT (cnn_layers=0): Global from start, like some AI models
- Hybrid (both>0): Biologically realistic, computationally efficient

This mirrors evolution: simple organisms use local processing,
complex mammals add high-level semantic areas.
"""

# -----------------------------
# Reusable Components (unchanged from previous version)
# -----------------------------
class SinusoidalPosEmb(nn.Module):
    """Timestep embedding for diffusion models"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)) 
        )
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ConvBlock(nn.Module):
    """
    Pure convolutional processing - analogous to V1/V2 visual cortex.
    
    Biological Analog:
    - V1/V2 neurons have small receptive fields
    - Process local features independently
    - Fast, efficient, massively parallel
    """
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.gelu(self.conv1(x))
        t_emb_spatial = rearrange('b c -> b c 1 1', self.time_mlp(t_emb))
        h = add('b c h w, b c 1 1', h, t_emb_spatial)
        h = F.gelu(self.conv2(h))
        return add('b c h w, b c h w', h, self.skip(x))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for semantic understanding (IT cortex analog)"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(
            'b (three heads head_dim) h w -> three b heads (h w) head_dim',
            qkv, three=3, heads=self.num_heads, head_dim=self.head_dim
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = dot('b heads n_pos d, b heads m_pos d -> b heads n_pos m_pos', q, k)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        out = dot('b heads n_pos m_pos, b heads m_pos d -> b heads n_pos d', attn, v)
        out = rearrange(
            'b heads (h w) head_dim -> b (heads head_dim) h w',
            out, heads=self.num_heads, head_dim=self.head_dim, h=H, w=W
        )
        return self.proj(out)

class AdaLN(nn.Module):
    """Adaptive Layer Normalization - modulates features based on timestep"""
    def __init__(self, channels, emb_dim):
        super().__init__()
        #self.norm = nn.GroupNorm(8, channels)
        #self.ada_mlp = nn.Linear(emb_dim, channels * 2)  # scale + shift

        self.norm = nn.GroupNorm(8, channels)
        self.ada_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, channels * 2)
        )

        assert isinstance(self.ada_mlp[-1].weight, torch.Tensor) # just to make VS Code not complain
        assert isinstance(self.ada_mlp[-1].bias, torch.Tensor) # just to make VS Code not complain

        # CRITICAL: Initialize to near-identity
        nn.init.zeros_(self.ada_mlp[-1].weight)  # ← Start with scale≈0, shift≈0
        nn.init.zeros_(self.ada_mlp[-1].bias)
    
    def forward(self, x, t_emb):
        # Normalize
        x_norm = self.norm(x)
        
        # Get adaptive parameters from timestep
        ada_params = self.ada_mlp(t_emb)  # [B, channels*2]
        scale, shift = ada_params.chunk(2, dim=1)  # [B, channels] each
        
        # Reshape for broadcasting
        scale = rearrange('b c -> b c 1 1', scale)
        shift = rearrange('b c -> b c 1 1', shift)

        assert isinstance(scale, torch.Tensor) # just to make VS Code not complain

        # Modulate: scale affects sensitivity, shift adds bias
        return x_norm * (1 + scale) + shift
    
class ViTBlock(nn.Module):
    """
    Vision Transformer block - analogous to IT (Inferotemporal) cortex.
    
    Stacking multiple ViT blocks = deeper semantic reasoning at same spatial scale.
    Like IT cortex: multiple processing stages for abstract understanding.
    """
    def __init__(self, in_channels, out_channels, emb_dim, num_heads=8, mlp_ratio=2.0, spatial_size=8):
        super().__init__()
        self.channel_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.time_mlp = nn.Linear(emb_dim, out_channels)
        
        self.pos_embed = nn.Parameter(
            torch.randn(1, out_channels, spatial_size, spatial_size) * 0.02
        )

        self.use_AdaLN = True
        if self.use_AdaLN:
            self.norm1 = AdaLN(out_channels, emb_dim)
            self.norm2 = AdaLN(out_channels, emb_dim)
        else: #group norm.
            self.norm1 = nn.GroupNorm(8, out_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
        self.attn = MultiHeadAttention(out_channels, num_heads)
        
        mlp_hidden = int(out_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, out_channels, 1)
        )

    def forward(self, x, t_emb):
        x = self.channel_proj(x)
        x = x + self.pos_embed
        # AdaLN modulates based on timestep
        if self.use_AdaLN:
            x = x + self.attn(self.norm1(x, t_emb))  # ← Pass t_emb to norm
            x = x + self.mlp(self.norm2(x, t_emb))   # ← Pass t_emb to norm
        else:
            t_emb_spatial = rearrange('b c -> b c 1 1', self.time_mlp(t_emb))
            x = add('b c h w, b c 1 1', x, t_emb_spatial)
            x = add('b c h w, b c h w', x, self.attn(self.norm1(x)))
            x = add('b c h w, b c h w', x, self.mlp(self.norm2(x)))
        return x


class SemanticInjector(nn.Module):
    """
    Injects global semantic understanding into local rendering.
    Models corticocortical feedback connections (IT → V1/V2).
    """
    def __init__(self, semantic_channels, target_channels):
        super().__init__()
        self.proj = nn.Conv2d(semantic_channels, target_channels, 1)
        
    def forward(self, decoder_features, semantic_features, target_size):
        semantic_proj = self.proj(semantic_features)
        semantic_upsampled = F.interpolate(
            semantic_proj, size=target_size, mode='bilinear', align_corners=False
        )
        return add('b c h w, b c h w', decoder_features, semantic_upsampled)


def calculate_channel_schedule(num_layers, base_ch, max_ch=512):
    """
    Calculate channel counts for each CNN layer.
    
    Uses quadratic growth: base_ch, base_ch * 4, base_ch * 9, ...
    Caps at max_ch to prevent memory explosion.
    
    Parameters:
    -----------
    num_layers : int
        Number of CNN layers (0 = no CNN)
    base_ch : int
        Base channel count (typically 64)
    max_ch : int
        Maximum channels (typically 512)
    
    Returns:
    --------
    list of int
        Channel count at each layer, e.g. [64, 144, 256, 400, 512]
    
    Examples:
    ---------
    >>> calculate_channel_schedule(4, 64)
    [64, 144, 256, 400]
    
    >>> calculate_channel_schedule(0, 64)
    []
    """
    if num_layers == 0:
        return []
    
    channels = []
    for i in range(num_layers):
        if i == 0:
            ch = base_ch
        else:
            ch = base_ch * (i + 1) ** 2
            ch = min(ch, max_ch)
        channels.append(ch)
    
    return channels

def calculate_channel_schedule(num_layers, base_ch, max_ch=512):
    """
    Calculate channel counts matching AdaptiveSemanticCoordinatorUNet.
    
    Uses the same growth pattern:
    - First layer: base_ch
    - Subsequent layers: base_ch * (i+1)^2, capped at max_ch
    
    Parameters:
    -----------
    num_layers : int
        Number of CNN layers
    base_ch : int
        Base channel count (typically 64)
    max_ch : int
        Maximum channels (typically 512)
    
    Returns:
    --------
    list of int
        Channel count at each layer
    
    Examples:
    ---------
    >>> calculate_channel_schedule(4, 64)
    [64, 64, 256, 512]
    
    >>> calculate_channel_schedule(5, 64)
    [64, 64, 256, 512, 512]
    
    >>> calculate_channel_schedule(1, 64)
    [64]
    
    >>> calculate_channel_schedule(0, 64)
    []
    """
    if num_layers == 0:
        return []
    
    if num_layers == 1:
        return [base_ch]
    
    channels = [base_ch]  # First layer
    
    # Subsequent layers with quadratic growth
    for i in range(num_layers - 1):
        ch = base_ch * (i + 1) ** 2
        ch = min(ch, max_ch)
        channels.append(ch)
    
    return channels

def make_patchify_layer(in_ch, out_ch, in_res, out_res):
    """
    Smart patchify layer - automatically determines stride based on resolutions.
    
    If in_res == out_res: Just channel projection (1×1 conv)
    If in_res > out_res: Strided convolution to downsample
    """
    if in_res == out_res:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1)
    else:
        stride = in_res // out_res
        return nn.Conv2d(in_ch, out_ch, kernel_size=stride, stride=stride)

def make_unpatchify_layer(in_ch, out_ch, in_res, out_res):
    """
    Smart unpatchify layer - inverse of patchify.
    
    If in_res == out_res: Just channel projection (1×1 conv)
    If in_res < out_res: Transposed convolution to upsample
    """
    if in_res == out_res:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1)
    else:
        stride = out_res // in_res
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=stride, stride=stride)


# -----------------------------
# CNN Encoder Module
# -----------------------------
class CNNEncoder(nn.Module):
    """
    CNN Encoder: Progressive downsampling with skip connections.
    
    Takes an image and produces:
    1. Encoded features at target resolution
    2. Skip features at each intermediate resolution
    
    Parameters:
    -----------
    in_channels : int
        Input channels (typically 3 for RGB)
    channel_schedule : list of int
        Channels at each layer, e.g. [64, 128, 256, 512]
    emb_dim : int
        Timestep embedding dimension
    
    Attributes:
    -----------
    out_channels : int
        Number of channels in final output
    num_layers : int
        Number of encoding layers
    
    Forward:
    --------
    Args:
        x : Tensor [B, in_channels, H, W]
        t_emb : Tensor [B, emb_dim]
    
    Returns:
        encoded : Tensor [B, out_channels, H/2^n, W/2^n]
        skip_features : list of Tensors at each resolution
    """
    def __init__(self, in_channels, channel_schedule, emb_dim):
        super().__init__()
        self.channel_schedule = channel_schedule
        self.num_layers = len(channel_schedule)
        self.out_channels = channel_schedule[-1] if channel_schedule else in_channels
        
        if self.num_layers == 0:
            # Pure ViT mode: no encoding layers
            self.blocks = nn.ModuleList()
            self.pools = nn.ModuleList()
            return
        
        # Build encoder blocks
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        current_ch = in_channels
        for out_ch in channel_schedule:
            self.blocks.append(ConvBlock(current_ch, out_ch, emb_dim))
            self.pools.append(nn.AvgPool2d(2))
            current_ch = out_ch
    
    def forward(self, x, t_emb):
        """
        Encode input through CNN layers.
        
        Returns:
        --------
        encoded : Final encoded features
        skip_features : List of features at each resolution (for skip connections)
        """
        skip_features = []
        h = x
        
        for block, pool in zip(self.blocks, self.pools):
            h = block(h, t_emb)
            skip_features.append(h)
            h = pool(h)
        
        return h, skip_features
    

# -----------------------------
# CNN Decoder Module
# -----------------------------
class CNNDecoder(nn.Module):
    """
    CNN Decoder: Progressive upsampling with skip connections and semantic injection.
    
    Mirrors the encoder structure, using skip connections from encoder
    and semantic guidance from ViT.
    
    Parameters:
    -----------
    channel_schedule : list of int
        Same schedule as encoder (will be processed in reverse)
    out_channels : int
        Final output channels (typically 3 for RGB)
    emb_dim : int
        Timestep embedding dimension
    semantic_channels : int
        Channels in semantic features from ViT
    
    Forward:
    --------
    Args:
        x : Tensor [B, C, H, W] - Output from ViT (unpatchified)
        skip_features : list of Tensors - From encoder, in forward order
        semantic_features : Tensor [B, semantic_channels, vit_res, vit_res]
        t_emb : Tensor [B, emb_dim]
    
    Returns:
        decoded : Tensor [B, out_channels, H_final, W_final]
    """
    def __init__(self, channel_schedule, out_channels, emb_dim, semantic_channels):
        super().__init__()
        self.channel_schedule = channel_schedule
        self.num_layers = len(channel_schedule)
        
        if self.num_layers == 0:
            # Pure ViT mode: no decoding layers
            self.ups = nn.ModuleList()
            self.blocks = nn.ModuleList()
            self.injectors = nn.ModuleList()
            self.output_proj = nn.Conv2d(out_channels, out_channels, 1)
            return
        
        # Build decoder blocks (in reverse order)
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.injectors = nn.ModuleList()
        
        # Reverse the channel schedule for decoding
        reversed_schedule = list(reversed(channel_schedule))
        
        for i, out_ch in enumerate(reversed_schedule):
            # Determine input channels for this decoder level
            if i == 0:
                # First decoder block receives from unpatchify
                in_ch = channel_schedule[-1]
            else:
                # Subsequent blocks receive from previous decoder
                in_ch = reversed_schedule[i - 1]
            
            # Upsampling layer
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            
            # Decoder block (concatenates with skip connection)
            self.blocks.append(ConvBlock(out_ch * 2, out_ch, emb_dim))
            
            # Semantic injector
            self.injectors.append(SemanticInjector(semantic_channels, out_ch))
        
        # Final output projection
        self.output_proj = nn.Conv2d(channel_schedule[0], out_channels, 1)
    
    def forward(self, x, skip_features, semantic_features, t_emb):
        """
        Decode features back to image resolution.
        
        Uses skip connections from encoder and semantic guidance from ViT
        at each resolution level.
        """
        h = x
        
        # Process in reverse order (deepest to shallowest)
        for i, (up, block, injector) in enumerate(zip(self.ups, self.blocks, self.injectors)):
            # Upsample
            h = up(h)
            
            # Get corresponding skip connection (from end of list backwards)
            if skip_features:
                skip_idx = len(skip_features) - 1 - i
                skip = skip_features[skip_idx]
                h = torch.cat([h, skip], dim=1)
            
            # Decode
            h = block(h, t_emb)
            
            # Inject semantic guidance
            h = injector(h, semantic_features, target_size=h.shape[-2:])
        
        # Final output projection
        return self.output_proj(h)



# -----------------------------
# Main Unified Architecture
# -----------------------------
class UnifiedSemanticCoordinatorUNet(nn.Module):
    """
    Unified Three-Stage Architecture: CNN Encoder → ViT Semantic → CNN Decoder
    
    Clean modular design with separate encoder/decoder classes and explicit
    channel schedules.
    
    Parameters:
    ===========
    img_size : int
        Input image size (assumes square images)
        
    cnn_layers : int
        Number of CNN encoder/decoder blocks
        - 0 = Pure ViT (no CNN processing)
        - 1+ = Hybrid architecture
        
    vit_layers : int
        Number of stacked ViT blocks at semantic resolution
        - 0 = Pure CNN (no global semantic processing)
        - 1+ = Semantic reasoning enabled
        
    cnn_resolution : int
        Target spatial resolution where CNN encoder ends
        - Must be >= vit_resolution
        - Example: 16 means CNN processes down to 16×16
        
    vit_resolution : int
        Target spatial resolution for ViT semantic processing
        - Must be <= cnn_resolution
        - Example: 8 means ViT processes at 8×8
        
    base_ch : int
        Base channel count for CNN (grows with depth)
        
    vit_ch : int
        Channel count for ViT blocks
        
    Architecture Examples:
    ======================
    
    Pure CNN:
        cnn_layers=4, vit_layers=0
        → Standard U-Net
        
    Pure ViT:
        cnn_layers=0, vit_layers=12
        → Original ViT-style
        
    Hybrid (default):
        cnn_layers=4, vit_layers=3
        → CNN local + ViT semantic
        
    Decoupled:
        cnn_layers=2, cnn_resolution=32, vit_resolution=8
        → CNN to 32×32, patchify to 8×8, ViT processes at 8×8
    """
    
    def __init__(self, 
                 img_size=128,
                 cnn_layers=4,
                 vit_layers=3, 
                 cnn_resolution=16,
                 vit_resolution=8,
                 base_ch=64,
                 vit_ch=256,
                 emb_dim=128,
                 in_channels=3,
                 out_channels=3,
                 num_heads=8):
        super().__init__()
        
        # Validate configuration
        assert vit_resolution <= cnn_resolution, \
            f"vit_resolution ({vit_resolution}) must be <= cnn_resolution ({cnn_resolution})"
        assert cnn_resolution % vit_resolution == 0, \
            f"cnn_resolution ({cnn_resolution}) must be divisible by vit_resolution ({vit_resolution})"
        
        if cnn_layers > 0:
            expected_cnn_res = img_size // (2 ** cnn_layers)
            assert expected_cnn_res == cnn_resolution, \
                f"With {cnn_layers} CNN layers, {img_size}×{img_size} reaches " \
                f"{expected_cnn_res}×{expected_cnn_res}, not {cnn_resolution}×{cnn_resolution}"
        
        self.img_size = img_size
        self.cnn_layers = cnn_layers
        self.vit_layers = vit_layers
        self.cnn_resolution = cnn_resolution
        self.vit_resolution = vit_resolution
        self.base_ch = base_ch
        self.vit_ch = vit_ch
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )
        
        # ====================================================================
        # Calculate channel schedule (all channel math in one place!)
        # ====================================================================
        
        channel_schedule = calculate_channel_schedule(cnn_layers, base_ch)
        print(f"Channel schedule: {channel_schedule}")
        
        # ====================================================================
        # STAGE 1: CNN Encoder
        # ====================================================================
        
        self.cnn_encoder = CNNEncoder(in_channels, channel_schedule, emb_dim)
        
        # ====================================================================
        # INTERFACE: CNN ↔ ViT (Patchify/Unpatchify)
        # ====================================================================
        
        encoder_out_ch = self.cnn_encoder.out_channels
        
        self.to_vit = make_patchify_layer(
            encoder_out_ch, vit_ch, cnn_resolution, vit_resolution
        )
        
        self.from_vit = make_unpatchify_layer(
            vit_ch, encoder_out_ch, vit_resolution, cnn_resolution
        )
        
        # ====================================================================
        # STAGE 2: ViT Semantic Processing
        # ====================================================================
        
        self.vit_blocks = nn.ModuleList([
            ViTBlock(vit_ch, vit_ch, emb_dim, num_heads=num_heads, 
                    spatial_size=vit_resolution)
            for _ in range(vit_layers)
        ])
        
        # ====================================================================
        # STAGE 3: CNN Decoder
        # ====================================================================
        
        self.cnn_decoder = CNNDecoder(channel_schedule, out_channels, emb_dim, vit_ch)
    
    def forward(self, x, t):
        """
        Unified forward pass - clean three-stage pipeline.
        
        No conditional branching - works for all modes (pure CNN, pure ViT, hybrid).
        """
        # Time embedding
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        t_emb = self.time_mlp(t)
        
        # ====================================================================
        # STAGE 1: CNN Encoder
        # ====================================================================
        
        h, skip_features = self.cnn_encoder(x, t_emb)
        
        # ====================================================================
        # INTERFACE: Patchify (CNN resolution → ViT resolution)
        # ====================================================================
        
        h = self.to_vit(h)
        
        # ====================================================================
        # STAGE 2: ViT Semantic Processing
        # ====================================================================
        
        semantic = h
        for vit_block in self.vit_blocks:
            semantic = vit_block(semantic, t_emb)
        
        # ====================================================================
        # INTERFACE: Unpatchify (ViT resolution → CNN resolution)
        # ====================================================================
        
        h = self.from_vit(semantic)
        
        # ====================================================================
        # STAGE 3: CNN Decoder
        # ====================================================================
        
        h = self.cnn_decoder(h, skip_features, semantic, t_emb)
        
        return h


# -----------------------------
# Factory Function with Smart Defaults
# -----------------------------
def create_semantic_coordinator(img_size=128, 
                                config='balanced',
                                cnn_layers=None,
                                vit_layers=None,
                                cnn_resolution=None,
                                vit_resolution=8):
    """
    Create Unified Semantic Coordinator with smart defaults.
    
    Parameters:
    -----------
    img_size : int
        Input image resolution
        
    config : str
        Preset configuration:
        - 'pure_cnn': Fast local processing only
        - 'lightweight': Minimal semantic processing
        - 'balanced': Good trade-off (default)
        - 'semantic_heavy': Deep semantic understanding
        - 'pure_vit': All attention, no convolution
        
    cnn_layers, vit_layers : int or None
        Override config presets with explicit values
        
    cnn_resolution : int or None
        Where CNN encoder stops (auto-calculated if None)
        
    vit_resolution : int
        Resolution for ViT semantic processing (default: 8)
    
    Examples:
    ---------
    # Balanced hybrid
    model = create_semantic_coordinator(128, 'balanced')
    
    # Pure ViT
    model = create_semantic_coordinator(128, 'pure_vit')
    
    # Custom: CNN to 32×32, then ViT at 8×8
    model = create_semantic_coordinator(
        img_size=128,
        cnn_layers=2,
        vit_layers=4,
        cnn_resolution=32,
        vit_resolution=8
    )
    """
    
    # Auto-calculate cnn_resolution if not provided
    if cnn_resolution is None:
        if config == 'pure_vit':
            cnn_resolution = img_size  # No CNN downsampling
        else:
            # Default: CNN reaches vit_resolution
            cnn_resolution = vit_resolution
    
    # Calculate default cnn_layers if not provided
    if config != 'pure_vit' and cnn_layers is None:
        default_cnn_layers = int(math.log2(img_size / cnn_resolution))
    else:
        default_cnn_layers = 0
    
    configs = {
        'pure_cnn': {
            'cnn_layers': default_cnn_layers if cnn_layers is None else cnn_layers,
            'vit_layers': 0,
            'base_ch': 64,
            'vit_ch': 256
        },
        'lightweight': {
            'cnn_layers': default_cnn_layers if cnn_layers is None else cnn_layers,
            'vit_layers': 2,
            'base_ch': 48,
            'vit_ch': 192
        },
        'balanced': {
            'cnn_layers': default_cnn_layers if cnn_layers is None else cnn_layers,
            'vit_layers': 3,
            'base_ch': 64,
            'vit_ch': 256
        },
        'semantic_heavy': {
            'cnn_layers': default_cnn_layers if cnn_layers is None else cnn_layers,
            'vit_layers': 6,
            'base_ch': 80,
            'vit_ch': 320
        },
        'pure_vit': {
            'cnn_layers': 0,
            'vit_layers': 12,
            'base_ch': 384,
            'vit_ch': 384
        }
    }
    
    # Get base config
    cfg = configs[config].copy()
    
    # Override with explicit parameters
    if cnn_layers is not None:
        cfg['cnn_layers'] = cnn_layers
    if vit_layers is not None:
        cfg['vit_layers'] = vit_layers
    
    return UnifiedSemanticCoordinatorUNet(
        img_size=img_size,
        cnn_resolution=cnn_resolution,
        vit_resolution=vit_resolution,
        **cfg
    )

#######################################################################3
#######################################################################3
#######################################################################3
# Below here is deprecated
#######################################################################3
#######################################################################3
#######################################################################3
# -----------------------------
# Main Adaptive Architecture
# -----------------------------
#@DeprecationWarning
class AdaptiveSemanticCoordinatorUNet(nn.Module):
    """
    Fully Parameterized Biologically-Inspired Architecture
    
    Uses loops to construct arbitrary depth encoder/decoder, enabling:
    - Pure CNN models (vit_layers=0)
    - Pure ViT models (cnn_layers=0)  
    - Hybrid models (both>0) - the biologically realistic case
    
    Parameters:
    ===========
    img_size : int
        Input image size (assumes square images)
        
    cnn_layers : int
        Number of CNN encoder blocks (V1/V2/V4 analog)
        - Each layer: pools 2×, doubles channels
        - More layers = deeper local processing before semantics
        - Rule of thumb: log2(img_size / semantic_resolution)
        - Examples:
          * 64→8: needs 3 layers
          * 128→8: needs 4 layers
          * 256→8: needs 5 layers
        - 0 = Pure ViT (process patches directly, like original ViT)
        
    vit_layers : int
        Number of stacked ViT blocks at semantic resolution (IT cortex analog)
        - All process at SAME resolution (no pooling between)
        - More layers = deeper semantic reasoning
        - Examples:
          * 0 = Pure CNN (no global understanding, fast)
          * 1-2 = Light semantic processing
          * 3-4 = Sweet spot (good understanding, efficient)
          * 6+ = Very deep reasoning (expensive)
          
    semantic_resolution : int
        Target spatial resolution for semantic processing
        - Default: 8 (64 positions, good memory/quality trade-off)
        - Smaller (4): Very coarse, only 16 positions
        - Larger (16): Fine-grained, 256 positions (4× memory)
        - Rule of thumb: Each position should cover ~16-32 pixels
        
    base_ch : int
        Base channel count (doubles with each CNN layer)
        
    Biological Mapping Examples:
    =============================
    
    Simple Organism (Pure CNN):
        cnn_layers=5, vit_layers=0
        → Only local processing, no global understanding
        → Fast, efficient, but may lack coherence
        
    Human-like (Hybrid):
        cnn_layers=4, vit_layers=3, semantic_resolution=8
        → V1/V2 local processing (4 CNN layers)
        → IT semantic understanding (3 ViT layers at 8×8)
        → Feedback to guide rendering
        → Biologically realistic, computationally efficient
        
    Pure Attention (Pure ViT):
        cnn_layers=0, vit_layers=12, semantic_resolution=16
        → Direct patch processing like original ViT
        → All global, no local processing stage
        → Expensive but maximally flexible
    
    Memory Implications:
    ====================
    
    Attention positions = (semantic_resolution)² × vit_layers
    
    Examples (with semantic_resolution=8):
        vit_layers=0:  0 attention ops (pure CNN)
        vit_layers=3:  192 attention ops (3 × 64)
        vit_layers=6:  384 attention ops (6 × 64)
        
    Compare to full ViT at 128×128 with patch_size=8:
        256 positions × 12 layers = 3,072 attention ops
        
    This architecture: 10-15× more efficient!
    
    Observed Training Dynamics (128×128, 4 CNN + 3 ViT):
    =====================================================
    - 44 minutes: Recognizable limbs (V4 working)
    - 5 hours: Balanced poses, appropriate interactions (IT working!)
    - Matches biological development: local features → semantics
    
    Neuroscience References:
    ========================
    - Felleman & Van Essen (1991): Visual hierarchy
    - DiCarlo & Cox (2007): IT cortex function
    - Gilbert & Li (2013): Feedback connections
    - Kriegeskorte (2015): CNNs as ventral stream models
    """
    
    def __init__(self, 
                 img_size=128,
                 cnn_layers=4,
                 vit_layers=3, 
                 semantic_resolution=8,
                 base_ch=64,
                 emb_dim=128,
                 in_channels=3,
                 out_channels=3,
                 num_heads=8):
        super().__init__()
        
        # Validate configuration
        if cnn_layers > 0:
            expected_resolution = img_size // (2 ** cnn_layers)
            assert expected_resolution == semantic_resolution, \
                f"With {cnn_layers} CNN layers, {img_size}×{img_size} input reaches " \
                f"{expected_resolution}×{expected_resolution}, not target {semantic_resolution}×{semantic_resolution}"
        
        self.img_size = img_size
        self.cnn_layers = cnn_layers
        self.vit_layers = vit_layers
        self.semantic_resolution = semantic_resolution
        self.base_ch = base_ch
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )
        
        # ====================================================================
        # ENCODER: Looped Construction
        # ====================================================================
        
        # Initial block at input resolution (provides skip connection for final decoder)
        self.input_block = ConvBlock(in_channels, base_ch, emb_dim)
        self.input_pool = nn.AvgPool2d(2)
        
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        self.encoder_configs = []  # Track (channels, resolution) for decoder
        
        # First config for input block
        self.encoder_configs.append({
            'channels': base_ch,
            'resolution': img_size,
            'index': -1  # Special index for input block
        })
        
        current_ch = base_ch
        current_res = img_size // 2  # After input_pool
        
        for i in range(cnn_layers - 1):  # -1 because input_block is first layer
            out_ch = base_ch * (2 ** (i + 1))
            out_ch = base_ch * (i + 1)**2 # Instead of esponentially growing channels quadratically
            if out_ch > 512:
                out_ch = 512
            self.encoder_blocks.append(ConvBlock(current_ch, out_ch, emb_dim))
            self.encoder_pools.append(nn.AvgPool2d(2))
            
            # Store config for decoder (before pooling)
            self.encoder_configs.append({
                'channels': out_ch,
                'resolution': current_res,
                'index': i
            })
            
            current_ch = out_ch
            current_res = current_res // 2
        
        # After CNN layers, we're at semantic_resolution
        self.semantic_channels = current_ch
        
        # ====================================================================
        # SEMANTIC PROCESSING: Stacked ViT blocks at same resolution
        # ====================================================================
        
        self.semantic_blocks = nn.ModuleList()
        for i in range(vit_layers):
            self.semantic_blocks.append(
                ViTBlock(self.semantic_channels, self.semantic_channels, 
                        emb_dim, num_heads=num_heads, spatial_size=semantic_resolution)
            )
        
        # ====================================================================
        # SEMANTIC INJECTION: One injector per decoder level
        # ====================================================================
        
        self.semantic_injectors = nn.ModuleList()
        for config in self.encoder_configs:
            self.semantic_injectors.append(
                SemanticInjector(self.semantic_channels, config['channels'])
            )
        
        # ====================================================================
        # DECODER: Mirror of encoder (reversed)
        # ====================================================================
        
        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        # Build decoder in reverse order (including initial block)
        for i in reversed(range(len(self.encoder_configs))):
            config = self.encoder_configs[i]
            
            # Upsample layer
            if i == len(self.encoder_configs) - 1:
                in_ch = self.semantic_channels
            else:
                in_ch = self.encoder_configs[i + 1]['channels']
            
            self.decoder_ups.append(
                nn.ConvTranspose2d(in_ch, config['channels'], kernel_size=2, stride=2)
            )
            
            # Decoder block (receives upsampled + skip connection)
            decoder_in_ch = config['channels'] * 2  # Concatenation
            decoder_out_ch = config['channels']
            self.decoder_blocks.append(
                ConvBlock(decoder_in_ch, decoder_out_ch, emb_dim)
            )
        
        # Output projection
        output_in_ch = base_ch  # First encoder config channels
        self.outc = nn.Conv2d(output_in_ch, out_channels, 1)

    def forward(self, x, t):
        """
        Forward pass with looped encoder/decoder construction.
        
        Flow:
        1. CNN encoder: Local feature extraction (V1/V2/V4)
        2. ViT semantic: Global understanding (IT cortex)
        3. Decoder with injection: Semantically-guided rendering
        """
        # Time embedding
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        t_emb = self.time_mlp(t)
        
        # ====================================================================
        # ENCODER: Loop through CNN layers
        # ====================================================================
        encoder_features = []
        
        # Initial block at input resolution
        h = self.input_block(x, t_emb)  # [b, 3, 128, 128] -> [b, 64, 128, 128]
        encoder_features.append(h)
        h = self.input_pool(h)  # [b, 64, 64, 64]
        
        # Remaining encoder blocks
        for i, (block, pool) in enumerate(zip(self.encoder_blocks, self.encoder_pools)):
            h = block(h, t_emb)
            encoder_features.append(h)  # Store for skip connections
            h = pool(h)
        
        # Now h is at semantic_resolution
        
        # ====================================================================
        # SEMANTIC PROCESSING: Loop through ViT layers (same resolution)
        # ====================================================================
        semantic = h
        for vit_block in self.semantic_blocks:
            semantic = vit_block(semantic, t_emb)
        
        # ====================================================================
        # DECODER: Loop through decoder layers (reverse order)
        # ====================================================================
        h = semantic
        
        for i, (up, block, injector) in enumerate(zip(
            self.decoder_ups, 
            self.decoder_blocks,
            reversed(self.semantic_injectors)
        )):
            # Upsample
            h = up(h)
            
            # Get corresponding encoder feature (reversed index)
            encoder_idx = len(encoder_features) - 1 - i
            skip = encoder_features[encoder_idx]
            
            # Concatenate with skip connection
            h = torch.cat([h, skip], dim=1)
            # Interesting experiment - disable skip connections here
            # h = torch.cat([h, torch.zeros_like(skip)], dim=1)
            # Decode
            h = block(h, t_emb)
            
            # Inject semantic guidance
            h = injector(h, semantic, target_size=h.shape[-2:])
        
        return self.outc(h)


# Factory function with smart defaults
def create_semantic_coordinator_old(img_size=128, 
                                config='balanced',
                                cnn_layers=None,
                                vit_layers=None,
                                semantic_resolution=8):
    """
    Create biologically-inspired Semantic Coordinator with smart defaults.
    
    Parameters:
    -----------
    img_size : int
        Input image resolution
        
    config : str
        Preset configuration:
        - 'pure_cnn': Fast local processing, no global understanding
        - 'lightweight': Minimal semantic processing
        - 'balanced': Good trade-off (default)
        - 'semantic_heavy': Deep semantic understanding
        - 'pure_vit': All attention, no convolution
        
    cnn_layers, vit_layers : int or None
        Override config presets with explicit values
        
    semantic_resolution : int
        Target resolution for semantic processing (default: 8)
    
    Examples:
    ---------
    # Use preset
    model = create_semantic_coordinator(128, 'balanced')
    
    # Custom configuration
    model = create_semantic_coordinator(256, cnn_layers=5, vit_layers=4)
    
    # Pure CNN for fast inference
    model = create_semantic_coordinator(128, 'pure_cnn')
    """
    
    # Smart defaults based on input size
    default_cnn_layers = int(math.log2(img_size / semantic_resolution))
    
    configs = {
        'pure_cnn': {
            'cnn_layers': default_cnn_layers,
            'vit_layers': 0,
            'base_ch': 64,
            'notes': """while the loss declines faster early 
                        in training, and looks noticably 
                        better in 5 hours of training,
                        by 12 hours losses are ~20% worse 
                        than 'bigger' and images are visibly worse
                    """
        },
        'lightweight': {
            'cnn_layers': default_cnn_layers,
            'vit_layers': 2,
            'base_ch': 48
        },
        'balanced': {
            'cnn_layers': default_cnn_layers,
            'vit_layers': 3,
            'base_ch': 64
        },
        'bigger': {
            'cnn_layers': default_cnn_layers,
            'vit_layers': 4,
            'base_ch': 64
        },
        'semantic_heavy': {
            'cnn_layers': default_cnn_layers,
            'vit_layers': 6,
            'base_ch': 80
        },
        'pure_vit': {
            'cnn_layers': 0,
            'vit_layers': 12,
            'base_ch': 96
        }
    }
    
    # Get base config
    cfg = configs[config].copy()
    
    # Override with explicit parameters
    if cnn_layers is not None:
        cfg['cnn_layers'] = cnn_layers
    if vit_layers is not None:
        cfg['vit_layers'] = vit_layers
    
    return AdaptiveSemanticCoordinatorUNet(
        img_size=img_size,
        semantic_resolution=semantic_resolution,
        **cfg
    )


# Testing and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Semantic Coordinator - Biologically Inspired")
    print("=" * 70)
    
    # Test different configurations
    configs_to_test = [
        ('pure_cnn', "Pure CNN (like simple organisms)"),
        ('balanced', "Hybrid CNN+ViT (like mammalian cortex)"),
        ('semantic_heavy', "Deep semantic reasoning"),
    ]
    
    for config_name, description in configs_to_test:
        print(f"\n{description}:")
        print(f"  Config: '{config_name}'")
        
        model = create_semantic_coordinator(128, config_name)
        
        print(f"  CNN layers: {model.cnn_layers}")
        print(f"  ViT layers: {model.vit_layers}")
        print(f"  Semantic resolution: {model.semantic_resolution}×{model.semantic_resolution}")
        
        # Calculate attention positions
        attn_positions = (model.semantic_resolution ** 2) * model.vit_layers
        print(f"  Attention positions: {attn_positions}")
        
        # Test forward pass
        x = torch.randn(2, 3, 128, 128)
        t = torch.randn(2)
        
        with torch.no_grad():
            out = model(x, t)
        
        print(f"  ✓ Forward pass: {x.shape} → {out.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 70)
    print("Resolution Adaptation Demo")
    print("=" * 70)
    
    for img_size in [64, 128, 256]:
        model = create_semantic_coordinator(img_size, 'balanced')
        print(f"\n{img_size}×{img_size} input:")
        print(f"  CNN layers: {model.cnn_layers} (auto-calculated)")
        print(f"  Reaches: {img_size // (2**model.cnn_layers)}×{img_size // (2**model.cnn_layers)}")
        print(f"  Semantic positions: {model.semantic_resolution ** 2}")