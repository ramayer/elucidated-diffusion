import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://chatgpt.com/c/6939a2b4-a380-8331-a461-9f760607922b
# It's the model after the comment
#    “please give me the fully fixed code including buffer registration and input projection”
# run with
#    model = MultiScaleViT_EDM(dim=256, heads=8, blocks_per_scale=4)
#
# It generates very nice textures appropriate to the source material; but
# does not do well on global consistency or composition across distant
# parts of an image.


# ------------------------------------------------------------
# Sinusoidal time embedding (classic diffusion style)
# ------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim)

        # Register frequency bands as buffer so they move with model
        half = dim // 2
        freq = torch.exp(
            torch.linspace(0, 6, half) * (torch.log(torch.tensor(10000.0)) / half)
        )
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, t):
        # t: (B,) or (B,1)
        t = t.view(-1, 1)  # (B,1)
        sinusoid = t * self.freq  # (B, half)
        emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1)

        emb = self.lin1(emb)
        emb = F.silu(emb)
        emb = self.lin2(emb)
        return emb  # (B, dim)


# ------------------------------------------------------------
# Lightweight Window Attention (with optional grid shifts)
# ------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift=False):
        super().__init__()
        self.dim = dim
        self.heads = num_heads
        self.window = window_size
        self.shift = shift

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.window

        # Optional shift (Swin-style)
        if self.shift:
            x = torch.roll(x, shifts=(-w // 2, -w // 2), dims=(2, 3))

        # Partition into windows
        assert H % w == 0 and W % w == 0
        x = x.reshape(B, C, H // w, w, W // w, w)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, nH, nW, w, w, C)
        windows = x.reshape(B * (H // w) * (W // w), w * w, C)

        out, _ = self.attn(windows, windows, windows)
        out = out.reshape(B, H // w, W // w, w, w, C)
        x = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

        # Undo shift
        if self.shift:
            x = torch.roll(x, shifts=(w // 2, w // 2), dims=(2, 3))

        return x


# ------------------------------------------------------------
# Simple 1×1 Conv "Feedforward" (MLP) for post-attention
# ------------------------------------------------------------

class ConvMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim * 4, 1)
        self.fc2 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


# ------------------------------------------------------------
# Transformer Block with Window Attention
# ------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window, shift):
        super().__init__()
        self.attn = WindowAttention(dim, num_heads, window, shift)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = ConvMLP(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # permute for LayerNorm
        h = x.permute(0, 2, 3, 1)
        h = self.norm1(h).permute(0, 3, 1, 2)
        x = x + self.attn(h)

        h = x.permute(0, 2, 3, 1)
        h = self.norm2(h).permute(0, 3, 1, 2)
        x = x + self.mlp(h)
        return x


# ------------------------------------------------------------
# Multi-Scale ViT Backbone (32 → 64 → 128)
# ------------------------------------------------------------

class MultiScaleViT(nn.Module):
    def __init__(self,
                 dim=256,
                 heads=8,
                 blocks_per_scale=4,
                 use_256=False):
        super().__init__()
        self.dim = dim
        self.blocks_per_scale = blocks_per_scale
        self.use_256 = use_256

        # ----------------------------------------------------
        # Fixed positional encoding (Fourier 2D)
        # ----------------------------------------------------
        pe = self._make_grid_positional_encoding(dim)
        self.register_buffer("pos_encoding_128", pe, persistent=False)

        # Input projection (3 → dim)
        self.in_proj = nn.Conv2d(3, dim, 1)

        # Time embedding
        self.time_embed = TimeEmbedding(dim)
        self.time_to_feat = nn.Linear(dim, dim)

        # 32×32 blocks
        self.down_128_to_32 = nn.Conv2d(dim, dim, 4, 4)  # 128→32
        self.block32 = nn.ModuleList([
            TransformerBlock(dim, heads, window=4, shift=(i % 2 == 1))
            for i in range(blocks_per_scale)
        ])

        # 64×64 blocks
        self.up_32_to_64 = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.block64 = nn.ModuleList([
            TransformerBlock(dim, heads, window=8, shift=(i % 2 == 1))
            for i in range(blocks_per_scale)
        ])

        # 128×128 blocks
        self.up_64_to_128 = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.block128 = nn.ModuleList([
            TransformerBlock(dim, heads, window=8, shift=(i % 2 == 1))
            for i in range(blocks_per_scale)
        ])

        # Optional 256×256 stage
        if use_256:
            self.up_128_to_256 = nn.ConvTranspose2d(dim, dim, 2, 2)
            self.block256 = nn.ModuleList([
                TransformerBlock(dim, heads, window=8, shift=(i % 2 == 1))
                for i in range(blocks_per_scale)
            ])

        # Final projection to RGB
        self.out_proj = nn.Conv2d(dim, 3, 1)

    # ------------------------------------------------------------
    # Make 2D Fourier positional encoding for 128×128
    # ------------------------------------------------------------
    def _make_grid_positional_encoding(self, dim):
        H = W = 128
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij"
        )
        scales = torch.exp(torch.linspace(0, 4, dim // 4))

        # Register freq bands
        self.register_buffer("pos_freq", scales, persistent=False)

        ys = y[..., None] * scales
        xs = x[..., None] * scales
        pe = torch.cat(
            [torch.sin(ys), torch.cos(ys), torch.sin(xs), torch.cos(xs)],
            dim=-1
        )  # (H,W,dim)

        return pe.permute(2, 0, 1).unsqueeze(0)  # (1,dim,H,W)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x, t):
        B, C, H, W = x.shape
        assert H == 128 and W == 128, "For training: expect 128×128 input"

        # Input projection
        x = self.in_proj(x)  # (B,dim,H,W)

        # Add fixed PE (resized if needed)
        pe = F.interpolate(self.pos_encoding_128,
                           size=(H, W),
                           mode="nearest")
        x = x + pe

        # Timestep embedding
        temb = self.time_embed(t)  # (B,dim)
        temb = self.time_to_feat(temb).view(B, self.dim, 1, 1)
        x = x + temb

        # ---- 32×32 ----
        x32 = self.down_128_to_32(x)
        for blk in self.block32:
            x32 = blk(x32)

        # ---- 64×64 ----
        x64 = self.up_32_to_64(x32)
        for blk in self.block64:
            x64 = blk(x64)

        # ---- 128×128 ----
        x128 = self.up_64_to_128(x64)
        for blk in self.block128:
            x128 = blk(x128)

        # ---- Optional 256×256 ----
        if self.use_256:
            x256 = self.up_128_to_256(x128)
            for blk in self.block256:
                x256 = blk(x256)
            return self.out_proj(x256)

        return self.out_proj(x128)


# ------------------------------------------------------------
# Wrapper that returns only predicted denoised image
# ------------------------------------------------------------

class MultiScaleViT_EDM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = MultiScaleViT(**kwargs)

    def forward(self, x, t):
        return self.model(x, t)  # only return image
