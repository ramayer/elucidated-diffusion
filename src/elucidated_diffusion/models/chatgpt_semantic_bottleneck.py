# https://chatgpt.com/share/693efecd-f3d8-800b-8573-193bc38f35fa

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Time Embedding (FiLM-style output)
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freq = torch.exp(
            torch.linspace(0, 6, half) * (torch.log(torch.tensor(10000.0)) / half)
        )
        self.register_buffer("freq", freq, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 2)  # scale + shift
        )

    def forward(self, t):
        t = t.view(-1, 1)
        sinusoid = t * self.freq
        emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1)
        return self.mlp(emb)  # (B, 2*dim)


# ============================================================
# Full Attention Block (Semantic Bottleneck)
# ============================================================

class FullAttentionBlock(nn.Module):
    """
    Used ONLY at 32×32.
    No windows, no ConvMLP.
    Forces global semantic binding.
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens = self.norm(tokens)
        out, _ = self.attn(tokens, tokens, tokens)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + out


# ============================================================
# Window Attention (unchanged, for texture stages)
# ============================================================

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.window = window
        self.shift = shift
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.window

        if self.shift:
            x = torch.roll(x, shifts=(-w // 2, -w // 2), dims=(2, 3))

        x = x.reshape(B, C, H // w, w, W // w, w)
        x = x.permute(0, 2, 4, 3, 5, 1)
        windows = x.reshape(-1, w * w, C)

        out, _ = self.attn(windows, windows, windows)

        out = out.reshape(B, H // w, W // w, w, w, C)
        x = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

        if self.shift:
            x = torch.roll(x, shifts=(w // 2, w // 2), dims=(2, 3))

        return x


# ============================================================
# Conv MLP (Texture Refinement)
# ============================================================

class ConvMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Transformer Block (Windowed, Texture Stage)
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.attn = WindowAttention(dim, heads, window, shift)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = ConvMLP(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(h)
        h = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.mlp(h)
        return x


# ============================================================
# Semantic Bottleneck ViT (EDM-Compatible)
# ============================================================

class SemanticBottleneckViT(nn.Module):
    def __init__(self, dim=256, heads=8, blocks_per_scale=4):
        super().__init__()
        self.dim = dim

        # Input / output
        self.in_proj = nn.Conv2d(3, dim, 1)
        self.out_proj = nn.Conv2d(dim, 3, 1)

        # Time conditioning
        self.time_embed = TimeEmbedding(dim)

        # Positional encoding (fixed Fourier)
        self.register_buffer("pos128", self._make_pe(128, dim), persistent=False)

        # Down / up
        self.down = nn.Conv2d(dim, dim, 4, 4)   # 128 → 32
        self.up32_64 = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.up64_128 = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.up128_256 = nn.ConvTranspose2d(dim, dim, 2, 2)

        # === SEMANTIC BOTTLENECK ===
        self.semantic_blocks = nn.ModuleList([
            FullAttentionBlock(dim, heads) for _ in range(2)
        ])

        # Projections for semantic skip injection
        self.sem_to_64 = nn.Conv2d(dim, dim, 1)
        self.sem_to_128 = nn.Conv2d(dim, dim, 1)
        self.sem_to_256 = nn.Conv2d(dim, dim, 1)

        # === TEXTURE STAGES ===
        self.blocks64 = nn.ModuleList([
            TransformerBlock(dim, heads, window=8, shift=i % 2)
            for i in range(blocks_per_scale)
        ])
        self.blocks128 = nn.ModuleList([
            TransformerBlock(dim, heads, window=8, shift=i % 2)
            for i in range(blocks_per_scale)
        ])
        self.blocks256 = nn.ModuleList([
            TransformerBlock(dim, heads, window=8, shift=i % 2)
            for i in range(blocks_per_scale)
        ])

    # --------------------------------------------------------

    def _make_pe(self, size, dim):
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size),
            torch.linspace(-1, 1, size),
            indexing="ij"
        )
        scales = torch.exp(torch.linspace(0, 4, dim // 4))
        ys = y[..., None] * scales
        xs = x[..., None] * scales
        pe = torch.cat(
            [torch.sin(ys), torch.cos(ys), torch.sin(xs), torch.cos(xs)],
            dim=-1
        )
        return pe.permute(2, 0, 1).unsqueeze(0)

    # --------------------------------------------------------

    def forward(self, x, t):
        B, _, H, W = x.shape
        assert H in (64, 128, 256)

        x = self.in_proj(x)

        # Positional encoding (interpolated)
        pe = F.interpolate(self.pos128, size=(H, W), mode="nearest")
        x = x + pe

        # Time FiLM
        tscale, tshift = self.time_embed(t).chunk(2, dim=1)
        x = x * (1 + tscale[:, :, None, None]) + tshift[:, :, None, None]

        # --- SEMANTIC 32×32 ---
        x32 = self.down(x)
        for blk in self.semantic_blocks:
            x32 = blk(x32)

        semantic = x32  # persistent semantic state

        # --- 64×64 ---
        if H >= 64:
            x64 = self.up32_64(x32)
            x64 = x64 + self.sem_to_64(F.interpolate(semantic, scale_factor=2))
            for blk in self.blocks64:
                x64 = blk(x64)

        # --- 128×128 ---
        if H >= 128:
            x128 = self.up64_128(x64)
            x128 = x128 + self.sem_to_128(F.interpolate(semantic, scale_factor=4))
            for blk in self.blocks128:
                x128 = blk(x128)

        # --- 256×256 ---
        if H == 256:
            x256 = self.up128_256(x128)
            x256 = x256 + self.sem_to_256(F.interpolate(semantic, scale_factor=8))
            for blk in self.blocks256:
                x256 = blk(x256)
            return self.out_proj(x256)

        return self.out_proj(x128 if H == 128 else x64)
