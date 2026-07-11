import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# See:  
# https://claude.ai/share/df352ae5-7dfa-4947-a677-c139035da2cb
# for many of the design decisions in this model.

try:
    from einx import dot, rearrange
except ImportError as e:
    raise ImportError("Please install einx: pip install einx") from e

# =============================================================================
# Time embedding
# =============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard transformer-style sinusoidal embedding of the diffusion timestep.

    The frequency bank here is tuned for inputs spanning roughly hundreds to
    thousands of units (e.g. DDPM-style integer timesteps in [0, 1000)).
    EDM-style noise conditioning (c_noise = sigma.log() / 4) is a continuous
    value with a much narrower range -- typically std ~0.3, span roughly
    [-1.6, 0.9] under common EDM hyperparameters (P_mean=-1.2, P_std=1.2).
    Fed directly into this embedding, that narrow range barely moves most of
    the frequency channels, so very different noise levels end up mapped to
    nearly identical embeddings (cosine similarity ~0.96-1.0 in practice) --
    the network loses most of its ability to tell noise levels apart.

    t_scale rescales the input before the frequency bank is applied, so the
    *effective* range seen by the embedding lands back in the well-separated
    regime this embedding was designed for, regardless of the raw input's
    native scale. Pass scale_for_std() a measurement of your timestep
    input's std (e.g. c_noise.std() over a batch) to compute a sensible
    fixed value -- fixed rather than learned, since the right order of
    magnitude is already computable from your own noise schedule's
    hyperparameters (P_mean/P_std), and a learned scale would need gradient
    signal flowing through the very embedding that starts out collapsed,
    which is a slow, easy-to-get-stuck starting point for exactly the
    quantity meant to fix that collapse.
    """

    def __init__(self, dim, t_scale=1.0):
        super().__init__()
        if dim < 4 or dim % 2 != 0:
            raise ValueError(f"emb_dim must be an even number >= 4, got {dim}")
        self.dim = dim
        self.t_scale = t_scale

    @staticmethod
    def scale_for_std(input_std, target_std=300.0):
        """
        Suggests a t_scale value given the std of your actual timestep input
        (e.g. c_noise.std() measured over a representative batch). Targets
        an effective post-scale std of ~300, comfortably in the
        well-separated regime DDPM-style integer timesteps already occupy.

        Example: under standard EDM defaults (P_mean=-1.2, P_std=1.2),
        c_noise has std = P_std / 4 = 0.3, so:
        >>> SinusoidalTimeEmbedding.scale_for_std(0.3)
        1000.0
        """
        return target_std / input_std

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        args = (t[:, None].float() * self.t_scale) * freqs[None, :]
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
    return _interpolate_schedule(num_layers, base_ch, top_ch, shape)


def _interpolate_schedule(num_layers, start_ch, end_ch, shape="linear"):
    """
    Channel counts smoothly interpolated from start_ch to end_ch over
    num_layers steps (inclusive of both ends), using the same curve
    shapes as channel_schedule. Unlike channel_schedule, the endpoints
    are hit exactly -- useful when a downstream shape (e.g. vit_ch at
    the ViT/decoder boundary) is a hard constraint rather than a cap.
    """
    if num_layers == 1:
        return [start_ch]

    t = torch.linspace(0, 1, num_layers)
    if shape == "linear":
        curve = t
    elif shape == "front_loaded":
        curve = t ** 0.5
    elif shape == "back_loaded":
        curve = t ** 2
    else:
        raise ValueError(f"Unknown shape '{shape}', expected linear/front_loaded/back_loaded")

    channels = start_ch + curve * (end_ch - start_ch)
    return [int(round(c.item())) for c in channels]


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
# Neural field decoder: a per-token-weight alternative to a single
# large-kernel transposed conv for going straight from the ViT's token
# grid to pixels (PixNerd-style, Wang et al. 2025).
# =============================================================================

class PixNerdStyleDecoder(nn.Module):
    """
    Replaces a single large-kernel ConvTranspose2d with a per-token neural
    field, following PixNerd (Wang et al. 2025, arXiv:2507.23268): each
    token's own feature vector generates the WEIGHTS of a small private
    2-layer MLP for that token's patch, rather than every patch sharing one
    global MLP. That MLP is then evaluated once per pixel in the patch,
    conditioned on the pixel's local coordinate AND the actual noisy pixel
    value at that position (not position alone).

    Two distinct problems this addresses, and why each part of the design
    is doing the work it does:

    1. Persistent per-pixel noise within a patch. A single shared MLP (the
       earlier NeuralFieldDecoder in this file) has to find one function
       that works for every token vector it might ever see -- a flat
       region needs that one shared function to behave correctly for that
       specific token among everything else it must also represent. Here,
       each token generates its OWN small MLP's weights directly, so
       precisely representing "this patch is flat" is a local, per-token
       fact the network can dial in directly, not something a single
       global function has to get right for every possible token
       simultaneously.

    2. Visible seams at patch boundaries. Position-based smoothing only
       guarantees continuity *within* a patch -- two physically adjacent
       pixels straddling a boundary belong to different tokens entirely,
       with no shared mechanism forcing their outputs to agree, even
       though a real image is continuous right across that boundary.
       Conditioning each pixel's prediction on the actual noisy pixel
       value (which IS continuous across the boundary, since the noise
       was added to a real image with no patch structure) gives the
       network a shared, patch-independent signal at every boundary --
       nearby pixels feed in nearby noisy values regardless of which
       patch they're nominally assigned to, which a model conditioned
       only on position-plus-token-vector never has access to.

    Row-wise normalization of the generated weights (dividing each output
    unit's weight vector by its own norm) is a stabilization trick from
    the same paper: weight-generator output magnitude can vary a lot
    during training, and normalizing keeps the per-token MLP's effective
    behavior numerically consistent regardless of how large the raw
    generated weights happen to be at any point in training.
    """

    def __init__(self, token_ch, out_channels, num_freqs=6, hidden_dim=64):
        super().__init__()
        self.num_freqs = num_freqs
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        coord_ch = num_freqs * 2 * 2  # x and y, each sincos-encoded
        self.input_dim = coord_ch + out_channels  # + the noisy pixel value itself

        w1_size = self.input_dim * hidden_dim
        b1_size = hidden_dim
        w2_size = hidden_dim * out_channels
        b2_size = out_channels
        self._sizes = (w1_size, b1_size, w2_size, b2_size)
        self.weight_gen = nn.Linear(token_ch, sum(self._sizes))

        self._coord_cache = {}  # (out_res, grid_res, device, dtype) -> (local_x, local_y)

    def _get_local_coords(self, out_res, grid_res, device, dtype):
        # Cache key includes device/dtype -- see the equivalent note on the
        # earlier coordinate-cache implementation this is modeled on: a
        # cache keyed only on (out_res, grid_res) would silently hold
        # stale-device tensors after a .to(device) call made after the
        # cache was first populated.
        key = (out_res, grid_res, device, dtype)
        if key not in self._coord_cache:
            patch_size = out_res // grid_res
            pixel_idx = torch.arange(out_res, device=device, dtype=dtype)
            local_pix = pixel_idx % patch_size
            denom = (patch_size - 1) / 2
            local_coord = (local_pix - denom) / (denom if denom > 0 else 1.0)
            local_x = local_coord[None, :].expand(out_res, out_res)
            local_y = local_coord[:, None].expand(out_res, out_res)
            self._coord_cache[key] = (local_x, local_y)
        return self._coord_cache[key]

    def _sincos(self, coord):
        freqs = 2.0 ** torch.arange(self.num_freqs, device=coord.device, dtype=coord.dtype)
        args = coord.unsqueeze(-1) * freqs * math.pi
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, tokens, noisy_pixels):
        """
        tokens: [B, token_ch, grid_res, grid_res] -- the ViT's output grid.
        noisy_pixels: [B, out_channels, out_res, out_res] -- the actual
            noisy input image this decoder is predicting from. out_res
            must be a multiple of grid_res.
        Returns: [B, out_channels, out_res, out_res].
        """
        B, _, gh, gw = tokens.shape
        out_res = noisy_pixels.shape[-1]
        if out_res % gh != 0:
            raise ValueError(f"out_res ({out_res}) must be divisible by grid_res ({gh})")
        patch_size = out_res // gh

        w1_size, b1_size, w2_size, b2_size = self._sizes
        gen = self.weight_gen(F.silu(rearrange("b c h w -> b h w c", tokens)))
        w1 = gen[..., :w1_size].reshape(B, gh, gw, self.hidden_dim, self.input_dim)
        b1 = gen[..., w1_size:w1_size + b1_size]
        w2 = gen[..., w1_size + b1_size:w1_size + b1_size + w2_size].reshape(
            B, gh, gw, self.out_channels, self.hidden_dim
        )
        b2 = gen[..., -b2_size:]

        w1 = w1 / (w1.norm(dim=-1, keepdim=True) + 1e-6)
        w2 = w2 / (w2.norm(dim=-1, keepdim=True) + 1e-6)

        # broadcast each token's generated weights to every pixel in its patch
        w1 = w1.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        b1 = b1.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        w2 = w2.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        b2 = b2.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)

        local_x, local_y = self._get_local_coords(out_res, gh, tokens.device, tokens.dtype)
        coord_enc = torch.cat([self._sincos(local_x), self._sincos(local_y)], dim=-1)
        coord_enc = rearrange("h w c -> b h w c", coord_enc, b=B)
        pixel_input = torch.cat([coord_enc, rearrange("b c h w -> b h w c", noisy_pixels)], dim=-1)

        h = F.silu(torch.einsum("bhwko,bhwo->bhwk", w1, pixel_input) + b1)
        out = torch.einsum("bhwok,bhwk->bhwo", w2, h) + b2
        return rearrange("b h w c -> b c h w", out)


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

    def forward(self, x, t_emb, level_injections=None):
        """
        level_injections: optional {level_index: tensor[B, channels[level_index], H, W]},
        added to that level's block output BEFORE it's stored as a skip
        connection and before pooling -- so an injected signal reaches both
        the rest of the encoder (via pooling onward) and the decoder (via
        the skip connection) from a single addition. Levels are indexed
        the same way channels/skip_features are: level 0 is the first,
        highest-resolution block. None (the default) reproduces the exact
        original forward pass -- existing callers/checkpoints are
        unaffected.
        """
        skip_features = []
        h = x
        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            h = block(h, t_emb)
            if level_injections is not None and i in level_injections:
                h = h + level_injections[i]
            skip_features.append(h)
            h = pool(h)
        return h, skip_features

class ThumbnailEncoder(nn.Module):
    def __init__(self, in_channels, hidden_ch, out_ch, emb_dim, num_layers=3):
        super().__init__()
        layers = []
        cur_ch = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(cur_ch, hidden_ch, 3, padding=1))
            layers.append(nn.GroupNorm(_largest_valid_group_count(hidden_ch, 8), hidden_ch))
            layers.append(nn.GELU())
            cur_ch = hidden_ch
        self.net = nn.Sequential(*layers)

        self.proj = nn.Conv2d(hidden_ch, out_ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Per-timestep gate rather than a single constant: how much the
        # thumbnail should be trusted plausibly differs by noise level (near-
        # total-noise steps have almost nothing else to go on; near-clean
        # steps already have most of the answer from x_t itself), so let
        # gradient descent find that curve instead of hand-picking one
        # fixed value. Zero-init exactly like AdaLN's to_scale_shift --
        # gate=0 for every t at the start of training, same as the plain
        # constant version, just now a function of t_emb rather than a
        # single number.
        self.to_gate = nn.Linear(emb_dim, out_ch)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.constant_(self.to_gate.bias, 1.0)   # was: nn.init.zeros_(self.to_gate.bias)

        # Dedicated null-path output, still bypassing proj AND gate
        # entirely -- "no thumbnail" shouldn't depend on t at all, and
        # shouldn't be reachable through a pathway that's optimizing for
        # the conditional case.
        #self.null_injection = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.register_buffer("null_injection", torch.zeros(1, out_ch, 1, 1))


    def forward(self, thumbnail, t_emb):
        gate = rearrange("b c -> b c 1 1", self.to_gate(t_emb))
        return self.proj(self.net(thumbnail)) * gate

    def null(self, batch_size, height, width, device, dtype):
        return self.null_injection.expand(batch_size, -1, height, width).to(device=device, dtype=dtype)

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

    Channel width shrinks from in_sem_ch to out_sem_ch as this happens, the
    same way the CNN's own channel schedule shrinks with resolution: fewer
    spatial positions at the bottleneck means each one has to represent more
    ("a green eye with bushy eyebrows"), while more positions near full
    resolution each cover less ground and don't need as wide a budget.

    upsample does the resolution AND channel change in one strided/transposed
    conv. fuse then mixes the (already spatially-positioned) semantic signal
    with this level's local decoder features -- a 1x1 conv is enough here,
    since fuse's job is combining channels at a position, not detecting
    spatial patterns; the semantic map arrives already smooth, and the
    decoder features arrive already locally processed by ConvBlock's own
    3x3 convs immediately before this runs.
    """

    def __init__(self, in_sem_ch, out_sem_ch, decoder_ch, in_res, out_res):
        super().__init__()
        self.upsample = make_resize_layer(in_sem_ch, out_sem_ch, in_res, out_res)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_sem_ch + decoder_ch, out_sem_ch, 1),
            nn.GELU(),
            nn.Conv2d(out_sem_ch, out_sem_ch, 1),
        )
        self.inject = nn.Conv2d(out_sem_ch, decoder_ch, 1)

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
    def __init__(
        self,
        channels,
        out_channels,
        emb_dim,
        vit_ch,
        entry_resolution,
        in_channels,
        semantic_min_ch=32,
        semantic_shape="linear",
        use_semantic_refinement=True,
    ):
        super().__init__()
        reversed_channels = list(reversed(channels))

        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.refiners = nn.ModuleList() if use_semantic_refinement else None

        # The semantic map shrinks in channel width across the same number
        # of levels the CNN decoder has, from vit_ch at the bottleneck down
        # to semantic_min_ch at full resolution. Its shape (linear /
        # front_loaded / back_loaded) is a free parameter, independent of
        # the CNN's own channel_shape -- there's no requirement the two
        # curves match, since a "how compressed is this concept" schedule
        # and a "how much raw texture capacity is needed here" schedule
        # are answering different questions. vit_ch must be hit exactly at
        # the bottleneck (it's a hard constraint, not a cap), so this uses
        # _interpolate_schedule rather than channel_schedule. num_levels+1
        # boundary points are generated (entry, then one exit per level) so
        # every level actually shrinks, rather than wasting the first level
        # on a vit_ch -> vit_ch no-op.
        #
        # When use_semantic_refinement is False (no ViT stage at all, i.e.
        # vit_layers=0), none of this is built -- no refiners, no vit_ch
        # -sized parameters anywhere in the decoder. The decoder falls back
        # to a plain skip-connection U-Net, since there's no semantic map
        # worth refining in the first place.
        if use_semantic_refinement:
            num_levels = len(reversed_channels)
            sem_boundaries = _interpolate_schedule(
                num_levels + 1, vit_ch, semantic_min_ch, shape=semantic_shape
            )

        # entry_resolution is cnn_resolution -- where both the CNN feature
        # path (post-bottleneck) and the semantic map (after being resized
        # from vit_resolution if the two differ) start from. Both then
        # double together at every decoder level up to img_size.
        res = entry_resolution
        for i, ch in enumerate(reversed_channels):
            in_ch = channels[-1] if i == 0 else reversed_channels[i - 1]
            self.ups.append(nn.ConvTranspose2d(in_ch, ch, kernel_size=2, stride=2))
            self.blocks.append(ConvBlock(ch * 2, ch, emb_dim))

            if use_semantic_refinement:
                in_sem_ch = sem_boundaries[i]
                out_sem_ch = sem_boundaries[i + 1]
                next_res = res * 2
                self.refiners.append(SemanticRefiner(in_sem_ch, out_sem_ch, ch, res, next_res))
            res = res * 2

        # With no CNN layers (pure ViT), the decoder loop above never runs,
        # so the final features are whatever from_vit produced, at
        # in_channels (== encoder.out_channels when there's no CNN at all).
        proj_in_ch = channels[0] if channels else in_channels
        self.output_proj = nn.Conv2d(proj_in_ch, out_channels, 1)

    def forward(self, x, skip_features, semantic_map, t_emb):
        h = x
        if self.refiners is not None:
            for up, block, refiner, skip in zip(
                self.ups, self.blocks, self.refiners, reversed(skip_features)
            ):
                h = up(h)
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h, semantic_map = refiner(h, semantic_map)
        else:
            for up, block, skip in zip(self.ups, self.blocks, reversed(skip_features)):
                h = up(h)
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
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
        Channel width of the ViT stage. Also the width of the semantic map
        at the decoder's bottleneck end.
    semantic_min_ch : int
        Width of the semantic map by the time it reaches the decoder's
        final (full-resolution) level. The map shrinks from vit_ch down to
        this value across the decoder, mirroring how the CNN's own channel
        budget shrinks as spatial resolution grows -- a coarse position can
        carry a compound concept ("a green eye with bushy eyebrows"), a
        fine position only needs to carry a narrower one ("green eye").
    semantic_shape : str
        'linear', 'front_loaded', or 'back_loaded' -- the curve the
        semantic channel shrinkage follows. Independent of channel_shape:
        there's no requirement that "how compressed a concept needs to be"
        and "how much raw texture capacity a resolution needs" follow the
        same curve.
    emb_dim : int
        Dimension of the timestep embedding.
    t_scale : float
        Rescales t before the sinusoidal frequency bank is applied. The
        default of 1.0 is correct for DDPM-style integer timesteps in
        roughly [0, 1000). For EDM-style continuous noise conditioning
        (passing c_noise = sigma.log() / 4 as t), c_noise's narrow native
        range (std ~0.3 under common EDM hyperparameters) leaves the
        embedding unable to distinguish noise levels well -- use
        SinusoidalTimeEmbedding.scale_for_std(c_noise.std()) to compute an
        appropriate value, or pass ~1000 as a reasonable starting point
        under typical EDM defaults (P_mean=-1.2, P_std=1.2).
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
        semantic_min_ch=32,
        semantic_shape="linear",
        emb_dim=128,
        t_scale=1.0,
        in_channels=3,
        out_channels=3,
        num_heads=8,
        max_ch=512,
        thumbnail_resolution=None,
        thumbnail_channels=3,
        thumbnail_hidden_ch=64,
        thumbnail_drop_prob=0.15,
    ):
        super().__init__()

        cnn_resolution = img_size // (2 ** cnn_layers)
        if vit_layers > 0 and cnn_resolution % vit_resolution != 0 and vit_resolution % cnn_resolution != 0:
            raise ValueError(
                f"cnn_resolution ({cnn_resolution}) and vit_resolution ({vit_resolution}) "
                "must evenly divide one another"
            )
        if vit_ch <= 0 and vit_layers > 0:
            raise ValueError(f"vit_ch must be a positive integer when vit_layers > 0, got {vit_ch}")

        self.img_size = img_size
        self.cnn_layers = cnn_layers
        self.vit_layers = vit_layers
        self.vit_resolution = vit_resolution
        self.vit_ch = vit_ch

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim, t_scale=t_scale),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

        channels = channel_schedule(cnn_layers, base_ch, max_ch=max_ch, shape=channel_shape)

        self.encoder = CNNEncoder(in_channels, channels, emb_dim)
        encoder_out_ch = self.encoder.out_channels

        # --- Thumbnail (SR) conditioning: optional, off by default ---
        # thumbnail_resolution=None (the default) builds nothing extra --
        # this class behaves exactly as before. Setting it wires up a
        # ThumbnailEncoder that injects additively into the one encoder
        # level whose spatial resolution matches thumbnail_resolution,
        # found automatically from img_size (never hardcoded), rather than
        # upsampling the thumbnail to img_size and concatenating at input.
        # That avoids two things: the network needing to learn around the
        # blur signature of an upsample it never needed to see, and the
        # noisy-input pathway (x_t at every t) picking up a shortcut for
        # gross structure that would otherwise need to come from x_t
        # itself. thumbnail_drop_prob controls how often, during training,
        # the real thumbnail is replaced by ThumbnailEncoder's learned null
        # signal instead -- the CFG-style dropout that keeps the backbone
        # practiced at inferring structure with no thumbnail at all.
        self.thumbnail_encoder = None
        self.thumbnail_inject_level = None
        self.thumbnail_drop_prob = thumbnail_drop_prob
        if thumbnail_resolution is not None:
            if cnn_layers == 0:
                raise ValueError("thumbnail conditioning needs cnn_layers > 0 (no encoder stage to inject into)")
            if img_size % thumbnail_resolution != 0:
                raise ValueError(f"img_size ({img_size}) must be divisible by thumbnail_resolution ({thumbnail_resolution})")
            ratio = img_size // thumbnail_resolution
            log_ratio = math.log2(ratio)
            if not log_ratio.is_integer():
                raise ValueError(f"img_size / thumbnail_resolution ({ratio}) must be a power of 2")
            inject_level = int(log_ratio)
            if not (0 <= inject_level < cnn_layers):
                raise ValueError(
                    f"thumbnail_resolution ({thumbnail_resolution}) implies encoder level "
                    f"{inject_level}, out of range for cnn_layers={cnn_layers} (valid: 0..{cnn_layers - 1})"
                )
            self.thumbnail_inject_level = inject_level
            self.thumbnail_encoder = ThumbnailEncoder(
                thumbnail_channels, thumbnail_hidden_ch, channels[inject_level],
                emb_dim
            )

        # With vit_layers=0 there's no semantic reasoning happening and
        # nothing for a bridge to feed -- skip building it entirely, so a
        # pure-CNN config carries zero vit_ch-sized parameters anywhere
        # (no to_vit/from_vit, no semantic_to_cnn_res, no refiner chain).
        # The decoder falls back to a plain skip-connection U-Net.
        self.use_vit = vit_layers > 0

        # With cnn_layers=0 AND vit_layers>0 (pure ViT), from_vit would have
        # to jump from vit_resolution straight to img_size in one large,
        # non-overlapping transposed conv -- the single step responsible
        # for the per-pixel noise this class's docs discuss. There's also
        # no CNN decoder afterward (output_proj would be a bare 1x1 conv)
        # to give the network any other chance to enforce neighbor
        # agreement. PixNerdStyleDecoder replaces that whole jump with a
        # per-token neural field conditioned on the actual noisy input
        # pixels, which is smooth across patch boundaries by construction
        # rather than needing every kernel weight to separately land on
        # the right value. Only relevant in this exact configuration --
        # with any CNN layers present, the existing decoder's overlapping
        # convs already handle this correctly.
        self.use_neural_field_decoder = self.use_vit and cnn_layers == 0
        if self.use_neural_field_decoder and in_channels != out_channels:
            raise ValueError(
                f"PixNerdStyleDecoder conditions on the noisy input pixels directly, "
                f"so in_channels ({in_channels}) must equal out_channels ({out_channels}) "
                f"in this configuration (cnn_layers=0, vit_layers>0)."
            )

        if self.use_vit:
            self.to_vit = make_resize_layer(encoder_out_ch, vit_ch, cnn_resolution, vit_resolution)
            if self.use_neural_field_decoder:
                self.from_vit = None
                self.semantic_to_cnn_res = None
                self.neural_field_decoder = PixNerdStyleDecoder(vit_ch, out_channels)
            else:
                self.from_vit = make_resize_layer(vit_ch, encoder_out_ch, vit_resolution, cnn_resolution)
                self.neural_field_decoder = None
                # Separate, channel-preserving resize for the semantic map
                # itself (vit_ch -> vit_ch), so it enters the decoder at
                # cnn_resolution -- the same resolution the CNN decoder path
                # starts from -- rather than at vit_resolution, which differs
                # whenever vit_resolution is set independently of
                # cnn_resolution. A no-op (1x1 conv) when the two resolutions
                # already match.
                self.semantic_to_cnn_res = make_resize_layer(vit_ch, vit_ch, vit_resolution, cnn_resolution)
        else:
            self.to_vit = None
            self.from_vit = None
            self.semantic_to_cnn_res = None
            self.neural_field_decoder = None

        self.vit_blocks = nn.ModuleList([
            ViTBlock(vit_ch, emb_dim, num_heads=num_heads) for _ in range(vit_layers)
        ])

        # No CNN decoder at all when using the neural field path -- it goes
        # straight from the ViT's token grid to pixels, so the usual
        # skip-connection CNN decoder (which needs encoder feature maps
        # that don't exist when cnn_layers=0 anyway) isn't built.
        self.decoder = None if self.use_neural_field_decoder else CNNDecoder(
            channels, out_channels, emb_dim, vit_ch, cnn_resolution, in_channels,
            semantic_min_ch=semantic_min_ch, semantic_shape=semantic_shape,
            use_semantic_refinement=self.use_vit,
        )
    def forward(self, x, t, thumbnail=None, return_attn=False):
        if t.dim() > 1:
            t = t.squeeze(-1)
        t_emb = self.time_embed(t)

        level_injections = None
        if self.thumbnail_encoder is not None:
            B = x.shape[0]
            res = self.img_size // (2 ** self.thumbnail_inject_level)
            drop = self.training and torch.rand(()) < self.thumbnail_drop_prob
            if thumbnail is not None and not drop:
                inject = self.thumbnail_encoder(thumbnail, t_emb)   # was: self.thumbnail_encoder(thumbnail)
            else:
                inject = self.thumbnail_encoder.null(B, res, res, device=x.device, dtype=x.dtype)
            level_injections = {self.thumbnail_inject_level: inject}

        h, skip_features = self.encoder(x, t_emb, level_injections=level_injections)
        # ...everything below this line is completely unchanged...

        if not self.use_vit:
            # No ViT stage at all: skip the bridge entirely and decode
            # directly from the CNN encoder's output, plain-U-Net style.
            out = self.decoder(h, skip_features, None, t_emb)
            if return_attn:
                return out, []
            return out

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

        if self.use_neural_field_decoder:
            out = self.neural_field_decoder(semantic, noisy_pixels=x)
            if return_attn:
                return out, attn_maps
            return out

        h = self.from_vit(semantic)
        # The decoder's progressive refinement chain starts from the same
        # resolution the CNN decoder path starts from (cnn_resolution), not
        # vit_resolution -- those differ whenever vit_resolution is set
        # independently of cnn_resolution. Resize the semantic map (still
        # at vit_ch channels) up to cnn_resolution so both decoder inputs
        # start from a consistent resolution.
        semantic_for_decoder = self.semantic_to_cnn_res(semantic)
        out = self.decoder(h, skip_features, semantic_for_decoder, t_emb)

        if return_attn:
            return out, attn_maps
        return out