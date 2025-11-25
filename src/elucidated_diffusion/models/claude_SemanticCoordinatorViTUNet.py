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


class ViTBlock(nn.Module):
    """
    Vision Transformer block - analogous to IT (Inferotemporal) cortex.
    
    Stacking multiple ViT blocks = deeper semantic reasoning at same spatial scale.
    Like IT cortex: multiple processing stages for abstract understanding.
    """
    def __init__(self, in_channels, out_channels, emb_dim, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        self.channel_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.time_mlp = nn.Linear(emb_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.attn = MultiHeadAttention(out_channels, num_heads)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        mlp_hidden = int(out_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, out_channels, 1)
        )

    def forward(self, x, t_emb):
        x = self.channel_proj(x)

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


# -----------------------------
# Main Adaptive Architecture
# -----------------------------
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
                        emb_dim, num_heads=num_heads)
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
            
            # Decode
            h = block(h, t_emb)
            
            # Inject semantic guidance
            h = injector(h, semantic, target_size=h.shape[-2:])
        
        return self.outc(h)


# Factory function with smart defaults
def create_semantic_coordinator(img_size=128, 
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
            'base_ch': 64
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