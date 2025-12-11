import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# -------------------------
# Helpers
# -------------------------
def exists(x):
    return x is not None

def default(val, d):
    return val if val is not None else d

# -------------------------
# Sinusoidal time embedding -> small MLP
# -------------------------
class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # t: (B,) float in whatever schedule units you use
        B = t.shape[0]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)  # (B, dim)

# -------------------------
# Continuous absolute 2D positional embedding (MLP)
# -------------------------
class ContinuousPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, device, h, w, batch=1):
        # produce (batch, h*w, dim)
        ys = torch.linspace(0.0, 1.0, steps=h, device=device)
        xs = torch.linspace(0.0, 1.0, steps=w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1)  # (h, w, 2)
        coords = coords.view(h * w, 2)
        pe = self.mlp(coords)  # (h*w, dim)
        pe = pe.unsqueeze(0).expand(batch, -1, -1)
        return pe  # (batch, h*w, dim)

# -------------------------
# Window helpers (partition / reverse) with padding
# -------------------------
def pad_to_multiple(x, pad_h, pad_w):
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h))  # pad W then H

def window_partition(x, window_size):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x_p = pad_to_multiple(x, pad_h, pad_w)
    Hp, Wp = x_p.shape[2], x_p.shape[3]
    x_ = x_p.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    x_ = x_.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, nH, nW, ws, ws, C
    windows = x_.view(-1, window_size * window_size, C)  # (B * nH * nW, ws*ws, C)
    return windows, Hp, Wp, pad_h, pad_w

def window_reverse(windows, window_size, Hp, Wp, B, C, pad_h, pad_w):
    # windows: (B*nH*nW, ws*ws, C)
    nH = Hp // window_size
    nW = Wp // window_size
    x = windows.view(B, nH, nW, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, Hp, Wp)
    if pad_h or pad_w:
        x = x[:, :, : Hp - pad_h, : Wp - pad_w]
    return x

# -------------------------
# Relative position bias table (Swin-style)
# -------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        # create index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, ws, ws
        coords_flatten = coords.reshape(2, -1)  # 2, ws*ws
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, ws*ws, ws*ws
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, ws*ws, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # ws*ws, ws*ws
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self):
        # returns (num_heads, ws*ws, ws*ws)
        table = self.bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )  # ws*ws, ws*ws, num_heads
        return table.permute(2, 0, 1).contiguous()

# -------------------------
# Windowed Multi-Head Self-Attention with optional shift
# -------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=4, shift_size=0, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.shift_size = shift_size
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.rel_pos = RelativePositionBias(window_size, num_heads)

    def forward(self, x, H, W):
        # x: (B, N, C) with N = H*W
        B, N, C = x.shape
        assert N == H * W
        # reshape to B, C, H, W for patch windows
        x_sp = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        # optional shift
        if self.shift_size > 0:
            x_sp = torch.roll(x_sp, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        # partition windows
        windows, Hp, Wp, pad_h, pad_w = window_partition(x_sp, self.window_size)  # (num_windows*B, ws*ws, C)
        # qkv
        qkv = self.qkv(windows)  # (num_windows*B, ws*ws, 3C)
        qkv = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # 3, numW*B, heads, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (numW*B, heads, ws*ws, head_dim)
        # attention
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (numW*B, heads, ws*ws, ws*ws)
        rel_bias = self.rel_pos()  # (heads, ws*ws, ws*ws)
        attn = attn + rel_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # (numW*B, heads, ws*ws, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], out.shape[2], -1)  # (numW*B, ws*ws, C)
        # merge windows back
        x_merged = window_reverse(out, self.window_size, Hp, Wp, B, C, pad_h, pad_w)  # (B, C, H, W) padded trimmed
        # reverse shift
        if self.shift_size > 0:
            x_merged = torch.roll(x_merged, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        x_flat = rearrange(x_merged, 'b c h w -> b (h w) c')
        x_flat = self.proj(x_flat)
        x_flat = self.proj_drop(x_flat)
        return x_flat

# -------------------------
# Simple MLP used in transformer block
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Swin-like Transformer block (pre-norm)
# -------------------------
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=4, shift=False, mlp_mult=4, dropout=0., attn_dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, window_size=window_size,
                                    shift_size=(window_size // 2) if shift else 0,
                                    attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=mlp_mult, dropout=dropout)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ff(self.norm2(x))
        return x

# -------------------------
# Global transformer (full attention) for 32x32 semantics
# -------------------------
class GlobalTransformer(nn.Module):
    def __init__(self, dim, depth=6, num_heads=8, mlp_mult=4, dropout=0., attn_dropout=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=attn_dropout),
                'norm2': nn.LayerNorm(dim),
                'ff': FeedForward(dim, mult=mlp_mult, dropout=dropout)
            }) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, C)
        for b in self.blocks:
            x_attn, _ = b['attn'](b['norm1'](x), b['norm1'](x), b['norm1'](x))
            x = x + x_attn
            x = x + b['ff'](b['norm2'](x))
        return self.norm(x)

# -------------------------
# Spatially-keyed Cross-Attention (fine queries coarse keys)
# -------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sem, pos_bias=None):
        # x: (B, Nf, C) fine tokens
        # sem: (B, Ns, C) semantic tokens (coarse)
        B, Nf, C = x.shape
        Ns = sem.shape[1]
        q = self.to_q(x).view(B, Nf, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, heads, Nf, hd
        kv = self.to_kv(sem).view(B, Ns, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # each: B, heads, Ns, hd
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # B, heads, Nf, Ns
        if exists(pos_bias):
            # pos_bias should be (B, Nf, Ns) or broadcastable
            attn = attn + pos_bias.unsqueeze(1)  # broadcast heads
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Nf, C)
        out = self.proj(out)
        out = self.dropout(out)
        return x + out

# -------------------------
# A single shared local transformer composed of SwinBlocks
# -------------------------
class SharedLocalTransformer(nn.Module):
    def __init__(self, dim, depth=4, num_heads=8, window_size=4, mlp_mult=4, dropout=0., attn_dropout=0.):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(SwinBlock(dim, num_heads=num_heads,
                                    window_size=window_size,
                                    shift=(i % 2 == 1),
                                    mlp_mult=mlp_mult,
                                    dropout=dropout,
                                    attn_dropout=attn_dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        # x: (B, H*W, C)
        for blk in self.blocks:
            x = blk(x, H, W)
        return self.norm(x)

# -------------------------
# S3D-ViT: full model
# -------------------------
class S3DViT(nn.Module):
    def __init__(
        self,
        *,
        base_dim=256,          # default small model to fit ~6GB
        num_heads=8,
        global_depth=6,
        local_depth=4,
        window_size=4,
        patch_map = {32:4, 64:8, 128:16},  # patch sizes per scale (keep tokens ~8x8)
        mlp_mult=4,
        dropout=0.,
        attn_dropout=0.
    ):
        super().__init__()
        self.base_dim = base_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.patch_map = patch_map

        # patch embedders: note: for higher res we accept 6 channels (image + upsampled prev)
        self.patch32  = nn.Conv2d(3, base_dim, kernel_size=patch_map[32], stride=patch_map[32])
        self.patch64  = nn.Conv2d(6, base_dim, kernel_size=patch_map[64], stride=patch_map[64])
        self.patch128 = nn.Conv2d(6, base_dim, kernel_size=patch_map[128], stride=patch_map[128])

        # output heads (conv transpose)
        self.head32  = nn.ConvTranspose2d(base_dim, 3, kernel_size=patch_map[32], stride=patch_map[32])
        self.head64  = nn.ConvTranspose2d(base_dim, 3, kernel_size=patch_map[64], stride=patch_map[64])
        self.head128 = nn.ConvTranspose2d(base_dim, 3, kernel_size=patch_map[128], stride=patch_map[128])

        # time + continuous pos emb
        self.time = TimeEmbed(base_dim)
        self.pos_emb = ContinuousPosEmb(base_dim)

        # semantic backbone (global full attention) for 32x32 tokens (we choose 8x8 tokens so N=64)
        self.semantic_backbone = GlobalTransformer(base_dim, depth=global_depth, num_heads=num_heads,
                                                  mlp_mult=mlp_mult, dropout=dropout, attn_dropout=attn_dropout)

        # shared local transformer for 64/128
        self.local_backbone = SharedLocalTransformer(base_dim, depth=local_depth, num_heads=num_heads,
                                                     window_size=window_size, mlp_mult=mlp_mult,
                                                     dropout=dropout, attn_dropout=attn_dropout)

        # cross-attention module (fine -> semantic), spatial gating is accomplished by adding a learned positional bias
        self.cross_attn = CrossAttention(base_dim, num_heads=num_heads, dropout=dropout)

        # small norms
        self.norm = nn.LayerNorm(base_dim)

        # initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if exists(m.bias):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if exists(m.bias):
                    nn.init.zeros_(m.bias)

    def tokens_from_patch(self, x, patch_conv):
        # x: (B, C, H, W)
        t = patch_conv(x)  # (B, D, h, w)
        h, w = t.shape[2], t.shape[3]
        tokens = rearrange(t, 'b c h w -> b (h w) c')
        return tokens, h, w

    def forward_stage(self, x_in, prev_out_up, t_emb, patch_conv, head_conv,
                      semantic_tokens, sem_h, sem_w, use_cross=True):
        """
        x_in: image at target res (B,3,H,W) or concatenated with prev out (B,6,H,W)
        prev_out_up: optionally previous stage upsampled image (B,3,H,W) - already concatenated if needed
        semantic_tokens: (B, Ns, C)
        sem_h, sem_w: semantics grid dims
        """
        # patch
        tokens, H, W = self.tokens_from_patch(x_in, patch_conv)  # (B, H*W, C)
        B = tokens.shape[0]
        # add time + absolute pos
        pos = self.pos_emb(tokens.device, H, W, batch=B)  # (B, H*W, C)
        tokens = tokens + t_emb[:, None, :] + pos

        # semantic cross-attention: upsample semantic tokens to match a coarse grid:
        if use_cross and exists(semantic_tokens):
            # We will expand semantic tokens spatially by nearest upsample to the grid of tokens.
            # sem tokens shape is (B, Ns, C) where Ns = sem_h * sem_w (e.g., 8x8)
            sem_up = self.upsample_semantics(semantic_tokens, sem_h, sem_w, H, W)  # (B, Ns_up, C)
            # optionally create a simple separable positional affinity: compute distances between fine tokens and sem_up tokens
            # We compute a soft biased positional term based on L2 distance in normalized coords.
            # coords_fine: (B, Nf, 2), coords_sem: (B, Ns_up, 2)
            coords_f = self._coords(tokens.device, H, W, batch=B)  # (B, Nf, 2)
            coords_s = self._coords(tokens.device, int(sem_h * (H//sem_h)), int(sem_w * (W//sem_w)), batch=B)  # approx
            # compute coarse distance-based bias (negative squared distance)
            # (B, Nf, Ns_up)
            dist = -torch.cdist(coords_f, coords_s, p=2).clamp(min=1e-6)
            pos_bias = dist  # can scale or projection if desired
            tokens = self.cross_attn(tokens, sem_up, pos_bias=pos_bias)
        # local refinement (shared)
        tokens = self.local_backbone(tokens, H, W)
        tokens = self.norm(tokens)
        # reconstruct pixels
        out_spatial = rearrange(tokens, 'b (h w) c -> b c h w', h=H, w=W)
        pixels = head_conv(out_spatial)  # (B, 3, H*patch, W*patch)
        return pixels, tokens, H, W

    def upsample_semantics(self, semantic_tokens, sem_h, sem_w, to_h, to_w):
        # semantic_tokens: (B, Ns, C) where Ns = sem_h * sem_w
        B, Ns, C = semantic_tokens.shape
        sem_sp = rearrange(semantic_tokens, 'b (h w) c -> b c h w', h=sem_h, w=sem_w)
        sem_up = F.interpolate(sem_sp, size=(to_h, to_w), mode='bilinear', align_corners=False)
        sem_up_flat = rearrange(sem_up, 'b c h w -> b (h w) c')
        return sem_up_flat

    def _coords(self, device, h, w, batch=1):
        ys = torch.linspace(0.0, 1.0, steps=h, device=device)
        xs = torch.linspace(0.0, 1.0, steps=w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).view(1, h * w, 2)  # (1, N, 2)
        return coords.expand(batch, -1, -1)  # (B, N, 2)

    def forward(self, x, t):
        """
        x: (B, 3, H, W) where H,W in {32,64,128} ideally
        t: (B,) float timesteps
        returns: predicted pixels at input resolution
        """
        B = x.shape[0]
        t_emb = self.time(t)  # (B, C)

        # Stage 0: 32x32 semantic backbone (global attention)
        x32 = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        tokens32, h32, w32 = self.tokens_from_patch(x32, self.patch32)  # tokens from 4x4 patches -> 8x8 grid
        pos32 = self.pos_emb(tokens32.device, h32, w32, batch=B)
        tokens32 = tokens32 + t_emb[:, None, :] + pos32
        sem_tokens = self.semantic_backbone(tokens32)  # (B, Ns, C) where Ns = 8*8
        sem_h, sem_w = h32, w32
        out32 = rearrange(sem_tokens, 'b (h w) c -> b c h w', h=sem_h, w=sem_w)
        out32 = self.head32(out32)  # (B,3,32,32)

        # If input is 32: return early
        if x.shape[-1] == 32:
            return out32#, sem_tokens

        # Stage 1: 64x64 (shared local refinement)
        x64 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        out32_up = F.interpolate(out32, size=(64, 64), mode='bilinear', align_corners=False)
        # concatenate noisy input and upsampled prior prediction
        x64_in = torch.cat([x64, out32_up], dim=1)  # (B,6,64,64)
        out64, tokens64, H64, W64 = self.forward_stage(
            x64_in, out32_up, t_emb,
            patch_conv=self.patch64, head_conv=self.head64,
            semantic_tokens=sem_tokens, sem_h=sem_h, sem_w=sem_w, use_cross=True
        )
        if x.shape[-1] == 64:
            return out64#, tokens64

        # Stage 2: 128x128 (shared local backbone reused)
        x128 = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        out64_up = F.interpolate(out64, size=(128, 128), mode='bilinear', align_corners=False)
        x128_in = torch.cat([x128, out64_up], dim=1)
        out128, tokens128, H128, W128 = self.forward_stage(
            x128_in, out64_up, t_emb,
            patch_conv=self.patch128, head_conv=self.head128,
            semantic_tokens=sem_tokens, sem_h=sem_h, sem_w=sem_w, use_cross=True
        )
        return out128#, tokens128

# -------------------------
# Example default instantiation tuned for ~6GB
# -------------------------
def create_default_s3dvit():
    # Default dims are modest: base_dim=256, depth small.
    return S3DViT(
        base_dim=256,
        num_heads=8,
        global_depth=6,
        local_depth=4,
        window_size=4,
        patch_map={32:4, 64:8, 128:16},
        mlp_mult=4,
        dropout=0.0,
        attn_dropout=0.0
    )

# Quick sanity check (dry run)
if __name__ == "__main__":
    model = create_default_s3dvit()
    x = torch.randn(2, 3, 128, 128)
    t = torch.tensor([10.0, 20.0])
    out, tokens = model(x, t)
    print("out shape:", out.shape)   # expected (2,3,128,128)
    print("tokens shape:", tokens.shape)  # expected (2, H*W, base_dim)
