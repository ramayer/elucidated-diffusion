# https://chatgpt.com/share/693efecd-f3d8-800b-8573-193bc38f35fa
#
# only works at 128x128
#
# Trains slowly, but may be one of the better vit-only attempts.
#
# Over 4 hours and no pokemon shapes.

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Time embedding (unchanged)
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freq = torch.exp(
            torch.linspace(0, 6, half) * (torch.log(torch.tensor(10000.0)) / half)
        )
        self.register_buffer("freq", freq, persistent=False)
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        t = t.view(-1, 1)
        emb = torch.cat([torch.sin(t * self.freq), torch.cos(t * self.freq)], dim=1)
        return self.lin2(F.silu(self.lin1(emb)))


# ============================================================
# Window Attention
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
            x = torch.roll(x, (-w // 2, -w // 2), dims=(2, 3))

        x = x.view(B, C, H // w, w, W // w, w)
        x = x.permute(0, 2, 4, 3, 5, 1)
        tokens = x.reshape(-1, w * w, C)

        out, _ = self.attn(tokens, tokens, tokens)
        out = out.view(B, H // w, W // w, w, w, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

        if self.shift:
            out = torch.roll(out, (w // 2, w // 2), dims=(2, 3))
        return out


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.attn = WindowAttention(dim, heads, window, shift)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim * 4, dim, 1),
        )

    def forward(self, x):
        h = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(h)
        h = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.mlp(h)


# ============================================================
# Global Token Block (cheap, O(HW·T))
# ============================================================

class GlobalTokenBlock(nn.Module):
    def __init__(self, dim, heads, num_tokens):
        super().__init__()
        self.norm_feat = nn.LayerNorm(dim)
        self.norm_tok = nn.LayerNorm(dim)
        self.attn_tok = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_feat = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, tokens):
        B, C, H, W = x.shape
        feat = x.flatten(2).transpose(1, 2)

        feat_n = self.norm_feat(feat)
        tok_n = self.norm_tok(tokens)

        tokens = tokens + self.attn_tok(tok_n, feat_n, feat_n)[0]
        feat = feat + self.attn_feat(feat_n, tokens, tokens)[0]

        x = feat.transpose(1, 2).view(B, C, H, W)
        return x, tokens


# ============================================================
# Memory-Reduced Global Semantic ViT
# ============================================================

class GlobalSemanticViT_EDM(nn.Module):
    def __init__(
        self,
        dim=256,
        heads=8,
        blocks_per_scale=4,
        num_global_tokens=4,
        max_res=128,
        token_scales=(32, 64),  # where tokens interact
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.token_scales = set(token_scales)

        self.in_proj = nn.Conv2d(3, dim, 1)
        self.out_proj = nn.Conv2d(dim, 3, 1)

        self.time_embed = TimeEmbedding(dim)
        self.time_to_feat = nn.Linear(dim, dim)

        self.global_tokens = nn.Parameter(
            torch.randn(num_global_tokens, dim) * 0.02
        )
        self.global_block = GlobalTokenBlock(dim, heads, num_global_tokens)

        self.down = nn.Conv2d(dim, dim, 4, 4)
        self.up1 = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.up2 = nn.ConvTranspose2d(dim, dim, 2, 2)
        if max_res == 256:
            self.up3 = nn.ConvTranspose2d(dim, dim, 2, 2)

        self.block32 = nn.ModuleList(
            TransformerBlock(dim, heads, 4, i % 2)
            for i in range(blocks_per_scale)
        )
        self.block64 = nn.ModuleList(
            TransformerBlock(dim, heads, 8, i % 2)
            for i in range(blocks_per_scale)
        )
        self.block128 = nn.ModuleList(
            TransformerBlock(dim, heads, 8, i % 2)
            for i in range(blocks_per_scale)
        )
        if max_res == 256:
            self.block256 = nn.ModuleList(
                TransformerBlock(dim, heads, 8, i % 2)
                for i in range(blocks_per_scale)
            )

    def maybe_tokens(self, x, tokens):
        if x.shape[-1] in self.token_scales:
            return self.global_block(x, tokens)
        return x, tokens

    def forward(self, x, t):
        B, _, H, W = x.shape
        assert H in (64, 128, 256)

        x = self.in_proj(x)
        temb = self.time_to_feat(self.time_embed(t)).view(B, self.dim, 1, 1)
        x = x + temb

        tokens = self.global_tokens.unsqueeze(0).expand(B, -1, -1)

        # 32
        x32 = self.down(x)
        for blk in self.block32:
            x32 = blk(x32)
        x32, tokens = self.maybe_tokens(x32, tokens)

        # 64
        x64 = self.up1(x32)
        for blk in self.block64:
            x64 = blk(x64)
        x64, tokens = self.maybe_tokens(x64, tokens)
        if H == 64:
            return self.out_proj(x64)

        # 128
        x128 = self.up2(x64)
        for blk in self.block128:
            x128 = blk(x128)
        x128, tokens = self.maybe_tokens(x128, tokens)
        if H == 128:
            return self.out_proj(x128)

        # 256
        x256 = self.up3(x128)
        for blk in self.block256:
            x256 = blk(x256)
        return self.out_proj(x256)
