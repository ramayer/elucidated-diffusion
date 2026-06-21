import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einx import dot, rearrange
except ImportError as e:
    raise ImportError("Please install einx: pip install einx") from e


# =============================================================================
# Time embedding
# =============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        if dim < 4 or dim % 2 != 0:
            raise ValueError(f"emb_dim must be an even number >= 4, got {dim}")
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# =============================================================================
# 2D sinusoidal positional grid (non-learned, resolution-independent)
# =============================================================================
def make_2d_sincos_grid(channels, height, width, device, dtype):
    """
    Builds a [1, channels, height, width] non-learned positional embedding.

    Half the channels encode row position, half encode column position,
    each with the standard sin/cos frequency bank. Because this is a closed
    -form function of (h, w), it can be generated at ANY resolution on
    demand -- there is no shape lock the way a learned nn.Parameter has.
    """
    if channels % 4 != 0:
        raise ValueError(f"channels must be divisible by 4 for 2D sincos, got {channels}")

    quarter = channels // 4
    freqs = torch.exp(
        torch.arange(quarter, device=device, dtype=dtype)
        * -(math.log(10000.0) / max(quarter - 1, 1))
    )

    rows = torch.arange(height, device=device, dtype=dtype)
    cols = torch.arange(width, device=device, dtype=dtype)

    row_args = rows[:, None] * freqs[None, :]  # [H, quarter]
    col_args = cols[:, None] * freqs[None, :]  # [W, quarter]

    row_emb = torch.cat([row_args.sin(), row_args.cos()], dim=-1)  # [H, channels/2]
    col_emb = torch.cat([col_args.sin(), col_args.cos()], dim=-1)  # [W, channels/2]

    row_emb = rearrange("h c -> 1 c h 1", row_emb).expand(1, -1, height, width)
    col_emb = rearrange("w c -> 1 c 1 w", col_emb).expand(1, -1, height, width)

    return torch.cat([row_emb, col_emb], dim=1)  # [1, channels, H, W]


def _largest_valid_group_count(channels, preferred):
    """
    GroupNorm requires num_channels % num_groups == 0. Channel schedules
    here are computed from smooth curves and rounded, so an arbitrary
    channel count (e.g. 175) won't always divide evenly by a fixed
    preferred group count (e.g. 8). Fall back to the largest divisor of
    channels that is <= preferred, so every layer gets a valid GroupNorm
    no matter what channel_schedule() produces.
    """
    preferred = min(preferred, channels)
    for g in range(preferred, 0, -1):
        if channels % g == 0:
            return g
    return 1


# =============================================================================
# Adaptive timestep conditioning (used by BOTH the CNN and ViT stages)
# =============================================================================
class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.

    Normalizes features, then rescales/shifts them using a timestep-dependent
    scale and shift. This lets t change the network's *processing regime*
    (what downstream layers see) rather than just adding a fixed offset
    after the fact -- the same mechanism DiT popularized for diffusion
    transformers, applied here uniformly across every conditioned block.

    Zero-initialized so the block starts training as a plain, untouched
    GroupNorm (scale=0, shift=0 -> identity modulation), giving a stable
    starting point that the network gradually learns to deviate from.
    """

    def __init__(self, channels, emb_dim, groups=8):
        super().__init__()
        groups = _largest_valid_group_count(channels, groups)
        self.norm = nn.GroupNorm(groups, channels)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, channels * 2),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x, t_emb):
        scale, shift = self.to_scale_shift(t_emb).chunk(2, dim=-1)
        scale = rearrange("b c -> b c 1 1", scale)
        shift = rearrange("b c -> b c 1 1", shift)
        return self.norm(x) * (1 + scale) + shift


# =============================================================================
# CNN stage building block (V1/V2-like local processing)
# =============================================================================
class ConvBlock(nn.Module):
    """
    Local convolutional processing block, conditioned on the timestep via
    AdaLN before each conv -- the same conditioning mechanism the ViT stage
    uses, so 't' changes the network's processing regime consistently
    throughout the whole model rather than via two different mechanisms.
    """

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = AdaLN(in_ch, emb_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaLN(out_ch, emb_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.gelu(self.norm1(x, t_emb)))
        h = self.conv2(F.gelu(self.norm2(h, t_emb)))
        return h + self.skip(x)


# =============================================================================
# Multi-head self-attention (IT-cortex-like global/semantic processing)
# =============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention over a spatial grid.

    Pass return_attn=True to additionally get back the raw attention
    weights [B, heads, N, N] for visualization (e.g. seeing which patches
    attend to which). This has no effect on the computed output and zero
    cost when left False -- it just means we keep a reference to a tensor
    we'd otherwise discard.
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, return_attn=False):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(
            "b (three heads head_dim) h w -> three b heads (h w) head_dim",
            qkv, three=3, heads=self.num_heads, head_dim=self.head_dim,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = dot("b heads n d, b heads m d -> b heads n m", q, k) * self.scale
        attn = F.softmax(attn_logits, dim=-1)

        out = dot("b heads n m, b heads m d -> b heads n d", attn, v)
        out = rearrange(
            "b heads (h w) head_dim -> b (heads head_dim) h w",
            out, heads=self.num_heads, head_dim=self.head_dim, h=H, w=W,
        )
        out = self.proj(out)

        if return_attn:
            # [B, heads, H*W, H*W] -> reshape query/key axes back to grid
            # form so callers can index attention by (row, col) directly.
            attn_map = rearrange(
                "b heads (qh qw) (kh kw) -> b heads qh qw kh kw",
                attn, qh=H, qw=W, kh=H, kw=W,
            )
            return out, attn_map
        return out, None


# =============================================================================
# ViT stage building block (IT-cortex-like semantic processing)
# =============================================================================
class ViTBlock(nn.Module):
    """
    A single transformer block operating on the semantic-resolution grid.

    Positional information comes from a non-learned 2D sin/cos grid rather
    than a learned parameter tensor. This removes the resolution lock a
    learned positional embedding would otherwise impose: the same block can
    run at any vit_resolution without a shape mismatch, since the grid is
    just a closed-form function of (h, w) computed fresh each forward pass.
    """

    def __init__(self, channels, emb_dim, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = AdaLN(channels, emb_dim)
        self.attn = MultiHeadAttention(channels, num_heads)
        self.norm2 = AdaLN(channels, emb_dim)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, channels, 1),
        )

    def forward(self, x, t_emb, pos_grid, return_attn=False):
        x = x + pos_grid
        attn_out, attn_map = self.attn(self.norm1(x, t_emb), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x, t_emb))
        return x, attn_map


# =============================================================================
# Channel schedule
# =============================================================================
def channel_schedule(num_layers, base_ch, max_ch=512, shape="linear"):
    """
    Channel count at each CNN encoder/decoder layer.

    Each resolution level trades spatial capacity for channel capacity as
    you go deeper: less spatial area to work with, so more channels are
    needed to hold the same amount of information. How fast that growth
    should happen is dataset dependent:

    - 'front_loaded': most growth happens early. Good when fine, high
      resolution texture detail matters a lot relative to how many
      distinct high-level structures need to be told apart (e.g.
      photoreal textures over a fairly constrained subject, like faces).
    - 'back_loaded': most growth happens late, near the bottleneck. Good
      when structural/categorical variety matters more than fine texture
      (e.g. distinguishing many distinct shapes/species).
    - 'linear': constant growth per layer. A reasonable, low-assumption
      default to start sweeping from.

    Examples
    --------
    >>> channel_schedule(4, 64, shape='linear')
    [64, 128, 192, 256]
    >>> channel_schedule(4, 64, shape='front_loaded')
    [64, 175, 221, 256]
    >>> channel_schedule(4, 64, shape='back_loaded')
    [64, 85, 149, 256]
    """
    if num_layers == 0:
        return []
    if num_layers == 1:
        return [min(base_ch, max_ch)]

    top_ch = min(base_ch * num_layers, max_ch)
    t = torch.linspace(0, 1, num_layers)

    if shape == "linear":
        curve = t
    elif shape == "front_loaded":
        curve = t ** 0.5
    elif shape == "back_loaded":
        curve = t ** 2
    else:
        raise ValueError(f"Unknown shape '{shape}', expected linear/front_loaded/back_loaded")

    channels = base_ch + curve * (top_ch - base_ch)
    return [min(int(round(c.item())), max_ch) for c in channels]


# =============================================================================
# Resolution-changing helper, shared by the CNN<->ViT bridge and the
# decoder's progressive semantic refinement path.
# =============================================================================
def make_resize_layer(in_ch, out_ch, in_res, out_res):
    """
    Picks the right op to go from (in_ch, in_res) to (out_ch, out_res):
      - same resolution -> plain 1x1 channel projection
      - downsampling    -> strided conv
      - upsampling      -> transposed conv
    """
    if in_res == out_res:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1)
    if in_res > out_res:
        stride = in_res // out_res
        return nn.Conv2d(in_ch, out_ch, kernel_size=stride, stride=stride)
    stride = out_res // in_res
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=stride, stride=stride)


# =============================================================================
# CNN encoder: progressive downsampling, V1/V2-like local processing
# =============================================================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels, channels, emb_dim):
        super().__init__()
        self.out_channels = channels[-1] if channels else in_channels
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        cur_ch = in_channels
        for out_ch in channels:
            self.blocks.append(ConvBlock(cur_ch, out_ch, emb_dim))
            self.pools.append(nn.AvgPool2d(2))
            cur_ch = out_ch

    def forward(self, x, t_emb):
        skip_features = []
        h = x
        for block, pool in zip(self.blocks, self.pools):
            h = block(h, t_emb)
            skip_features.append(h)
            h = pool(h)
        return h, skip_features


# =============================================================================
# Progressive semantic refinement (corticocortical feedback, IT -> V1/V2)
# =============================================================================
class SemanticRefiner(nn.Module):
    """
    Carries the ViT's semantic understanding back down through the decoder,
    sharpening it at each step instead of upsampling once from the
    coordinator's native resolution to the full image size.

    At vit_resolution, the semantic map can only say something coarse like
    "this region is a green cat eye." A single bilinear upsample straight to
    image resolution just blurs that coarse statement across a big patch --
    it never gets more specific. Refining progressively, once per decoder
    level, lets each step use that level's own (higher-resolution) decoder
    features to sharpen the semantic statement -- "the upper half of this
    region is iris, the lower half is eyelid" -- before passing it on to be
    sharpened again at the next, even higher resolution.
    """

    def __init__(self, semantic_ch, decoder_ch, in_res, out_res):
        super().__init__()
        self.upsample = make_resize_layer(semantic_ch, semantic_ch, in_res, out_res)
        self.fuse = nn.Sequential(
            nn.Conv2d(semantic_ch + decoder_ch, semantic_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(semantic_ch, semantic_ch, 3, padding=1),
        )
        self.inject = nn.Conv2d(semantic_ch, decoder_ch, 1)

    def forward(self, decoder_features, semantic_map):
        semantic_map = self.upsample(semantic_map)
        refined = self.fuse(torch.cat([semantic_map, decoder_features], dim=1))
        decoder_features = decoder_features + self.inject(refined)
        return decoder_features, refined


# =============================================================================
# CNN decoder: progressive upsampling with skip connections and
# progressively-refined semantic guidance
# =============================================================================
class CNNDecoder(nn.Module):
    def __init__(self, channels, out_channels, emb_dim, semantic_ch, vit_resolution, in_channels):
        super().__init__()
        reversed_channels = list(reversed(channels))

        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.refiners = nn.ModuleList()

        res = vit_resolution
        for i, ch in enumerate(reversed_channels):
            in_ch = channels[-1] if i == 0 else reversed_channels[i - 1]
            self.ups.append(nn.ConvTranspose2d(in_ch, ch, kernel_size=2, stride=2))
            self.blocks.append(ConvBlock(ch * 2, ch, emb_dim))
            next_res = res * 2
            self.refiners.append(SemanticRefiner(semantic_ch, ch, res, next_res))
            res = next_res

        # With no CNN layers (pure ViT), the decoder loop above never runs,
        # so the final features are whatever from_vit produced, at
        # in_channels (== encoder.out_channels when there's no CNN at all).
        proj_in_ch = channels[0] if channels else in_channels
        self.output_proj = nn.Conv2d(proj_in_ch, out_channels, 1)

    def forward(self, x, skip_features, semantic_map, t_emb):
        h = x
        for up, block, refiner, skip in zip(
            self.ups, self.blocks, self.refiners, reversed(skip_features)
        ):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            h, semantic_map = refiner(h, semantic_map)
        return self.output_proj(h)


# =============================================================================
# Full model: CNN encoder -> ViT semantic stage -> CNN decoder with
# progressive semantic refinement
# =============================================================================
class CorticalRefinerUNet(nn.Module):
    """
    Three-stage diffusion backbone: a CNN encoder for local (V1/V2-like)
    processing, a ViT bottleneck for global semantic (IT-like) reasoning,
    and a CNN decoder that reconstructs the image while progressively
    sharpening the semantic signal as corticocortical feedback at every
    resolution it passes through on the way back up.

    Set cnn_layers=0 for a pure ViT model, vit_layers=0 for a pure CNN
    U-Net, or both > 0 for the hybrid.

    Parameters
    ----------
    img_size : int
        Input/output spatial resolution (square images).
    cnn_layers : int
        Number of CNN encoder/decoder levels.
    vit_layers : int
        Number of stacked transformer blocks at the semantic bottleneck.
    vit_resolution : int
        Spatial resolution at which the ViT stage operates. img_size must
        be divisible by 2 ** cnn_layers, and the result must be divisible
        by vit_resolution.
    base_ch : int
        Channel count at the first (highest-resolution) CNN level.
    channel_shape : str
        'linear', 'front_loaded', or 'back_loaded' -- see channel_schedule().
    vit_ch : int
        Channel width of the ViT stage.
    emb_dim : int
        Dimension of the timestep embedding.
    num_heads : int
        Attention heads in each ViT block.
    """

    def __init__(
        self,
        img_size=128,
        cnn_layers=4,
        vit_layers=3,
        vit_resolution=8,
        base_ch=64,
        channel_shape="linear",
        vit_ch=256,
        emb_dim=128,
        in_channels=3,
        out_channels=3,
        num_heads=8,
        max_ch=512,
    ):
        super().__init__()

        cnn_resolution = img_size // (2 ** cnn_layers)
        if cnn_resolution % vit_resolution != 0 and vit_resolution % cnn_resolution != 0:
            raise ValueError(
                f"cnn_resolution ({cnn_resolution}) and vit_resolution ({vit_resolution}) "
                "must evenly divide one another"
            )

        self.img_size = img_size
        self.cnn_layers = cnn_layers
        self.vit_layers = vit_layers
        self.vit_resolution = vit_resolution
        self.vit_ch = vit_ch

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

        channels = channel_schedule(cnn_layers, base_ch, max_ch=max_ch, shape=channel_shape)

        self.encoder = CNNEncoder(in_channels, channels, emb_dim)
        encoder_out_ch = self.encoder.out_channels

        self.to_vit = make_resize_layer(encoder_out_ch, vit_ch, cnn_resolution, vit_resolution)
        self.from_vit = make_resize_layer(vit_ch, encoder_out_ch, vit_resolution, cnn_resolution)

        self.vit_blocks = nn.ModuleList([
            ViTBlock(vit_ch, emb_dim, num_heads=num_heads) for _ in range(vit_layers)
        ])

        self.decoder = CNNDecoder(channels, out_channels, emb_dim, vit_ch, vit_resolution, in_channels)

    def forward(self, x, t, return_attn=False):
        if t.dim() > 1:
            t = t.squeeze(-1)
        t_emb = self.time_embed(t)

        h, skip_features = self.encoder(x, t_emb)
        h = self.to_vit(h)

        pos_grid = make_2d_sincos_grid(
            self.vit_ch, h.shape[-2], h.shape[-1], device=h.device, dtype=h.dtype
        )

        attn_maps = [] if return_attn else None
        semantic = h
        for block in self.vit_blocks:
            semantic, attn_map = block(semantic, t_emb, pos_grid, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn_map)

        h = self.from_vit(semantic)
        out = self.decoder(h, skip_features, semantic, t_emb)

        if return_attn:
            return out, attn_maps
        return out
