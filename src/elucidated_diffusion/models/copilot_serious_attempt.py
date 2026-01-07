import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Time embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dim() > 1:
            t = t.squeeze(-1)
        device = t.device
        half_dim = self.dim // 2
        emb_scale = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * freq[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class TimeMLP(nn.Module):
    def __init__(self, emb_dim, model_dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, t):
        return self.net(t.float())

# --- ResBlock with FiLM ---
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        scale, shift = self.time_proj(t_emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return self.skip(x) + h

# --- Self-attention block ---
class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t):
            return t.view(B, self.num_heads, C // self.num_heads, H * W)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)

# --- Full unrolled U-Net ---
class UNet128_WithStructureMap(nn.Module):
    def __init__(self, in_channels=3, base_ch=128, time_dim=256):
        super().__init__()
        self.time_mlp = TimeMLP(emb_dim=time_dim, model_dim=time_dim)

        # --- Down path ---
        self.enc0 = nn.Conv2d(in_channels, base_ch, 3, padding=1)  # 128x128 → 128x128, 128ch

        self.down1 = nn.ModuleList([
            ResBlock(base_ch, base_ch, time_dim),
            ResBlock(base_ch, base_ch, time_dim),
        ])
        self.downsample1 = nn.Conv2d(base_ch, 256, 3, stride=2, padding=1)  # → 64x64

        self.down2 = nn.ModuleList([
            ResBlock(256, 256, time_dim),
            ResBlock(256, 256, time_dim),
        ])
        self.downsample2 = nn.Conv2d(256, 384, 3, stride=2, padding=1)  # → 32x32

        self.down3 = nn.ModuleList([
            ResBlock(384, 384, time_dim),
            AttentionBlock(384),
            ResBlock(384, 384, time_dim),
        ])
        self.downsample3 = nn.Conv2d(384, 512, 3, stride=2, padding=1)  # → 16x16

        self.down4 = nn.ModuleList([
            ResBlock(512, 512, time_dim),
            AttentionBlock(512),
            ResBlock(512, 512, time_dim),
        ])
        self.downsample4 = nn.Conv2d(512, 512, 3, stride=2, padding=1)  # → 8x8

        # --- Structure map head ---
        self.struct_head = nn.Sequential(
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 32, 1)
        )
        # Predicts a 5-channel structure map at 16×16:
        # [0] head, [1] eyes, [2] nose, [3] mouth, [4] body/core
        # These are soft spatial maps (not hard labels)
        # Changed from 5 to 32 

        # --- Mid ---
        self.mid = nn.ModuleList([
            ResBlock(512, 512, time_dim),
            AttentionBlock(512),
            ResBlock(512, 512, time_dim),
        ])

        # --- Up path ---
        self.upsample4 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)  # 8x8 → 16x16
        self.up4 = nn.ModuleList([
            ResBlock(512 + 512 + 32, 512, time_dim),  # concat x4 + struct_map
            AttentionBlock(512),
            ResBlock(512, 512, time_dim),
        ])

        self.upsample3 = nn.ConvTranspose2d(512, 384, 4, stride=2, padding=1)  # 16x16 → 32x32
        self.up3 = nn.ModuleList([
            ResBlock(384 + 384, 384, time_dim),
            AttentionBlock(384),
            ResBlock(384, 384, time_dim),
        ])

        self.upsample2 = nn.ConvTranspose2d(384, 256, 4, stride=2, padding=1)  # 32x32 → 64x64
        self.up2 = nn.ModuleList([
            ResBlock(256 + 256, 256, time_dim),
            ResBlock(256, 256, time_dim),
        ])

        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 64x64 → 128x128
        self.up1 = nn.ModuleList([
            ResBlock(128 + 128, 128, time_dim),
            ResBlock(128, 128, time_dim),
        ])

        self.out_norm = nn.GroupNorm(32, 128)
        self.out_conv = nn.Conv2d(128, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Down path
        x0 = self.enc0(x)  # 128x128, 128
        h = x0
        for block in self.down1:
            h = block(h, t_emb)
        x1 = h

        h = self.downsample1(x1)  # 64x64, 256
        for block in self.down2:
            h = block(h, t_emb)
        x2 = h

        h = self.downsample2(x2)  # 32x32, 384
        for block in self.down3:
            h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)
        x3 = h

        h = self.downsample3(x3)  # 16x16, 512
        for block in self.down4:
            h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)
        x4 = h  # 16×16, 512

        # Predict structure map from x4
        struct_map = self.struct_head(x4)  # 16×16, 5 (now 32)
        # This is a soft spatial map indicating where key parts (head, eyes, etc.) are likely to be

        h = self.downsample4(x4)
        for block in self.mid:
            h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)
        x5 = h  # 8×8, 512

        # Up path
        h = self.upsample4(x5)  # 16×16, 512
        h = torch.cat([h, x4, struct_map], dim=1)  # 512 + 512 + 5 (now 32) = 1029
        for block in self.up4:
            h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)

        h = self.upsample3(h)  # 32×32, 384
        h = torch.cat([h, x3], dim=1)
        for block in self.up3:
            h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)

        h = self.upsample2(h)  # 64×64, 256
        h = torch.cat([h, x2], dim=1)
        for block in self.up2:
            h = block(h, t_emb)

        h = self.upsample1(h)  # 128×128, 128
        h = torch.cat([h, x1], dim=1)
        for block in self.up1:
            h = block(h, t_emb)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)

