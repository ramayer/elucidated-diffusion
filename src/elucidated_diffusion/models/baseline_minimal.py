# https://copilot.microsoft.com/conversations/join/9ipccWx6NaA8GBuwyMVQN
# https://copilot.microsoft.com/shares/u3Uih6Gc4DQfRAZQuYeQ4
# https://copilot.microsoft.com/shares/WACJmRo9dtiiyB9BEgiHF
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Sinusoidal time embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.dim() > 1:
            x = x.squeeze(-1)
        device = x.device
        half_dim = self.dim // 2
        emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb_freq = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb_freq[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# --- Single-conv ResBlock with additive skip ---
class ResBlock(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.time_mlp = nn.Linear(emb_dim, ch)

    def forward(self, x, t_emb):
        h = self.conv(x) + self.time_mlp(t_emb)[:, :, None, None]
        return x + F.relu(h)

# --- Minimal U-Net ---
class MinimalUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, emb_dim=128):
        super().__init__()
        self.time_mlp = SinusoidalPosEmb(emb_dim)

        # Down path
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, emb_dim)
        self.down2 = ResBlock(base_ch, emb_dim)
        self.down3 = ResBlock(base_ch, emb_dim)
        self.down4 = ResBlock(base_ch, emb_dim)
        self.pool = nn.AvgPool2d(2)

        # Mid
        self.mid = ResBlock(base_ch, emb_dim)

        # Up path
        self.up4 = ResBlock(base_ch, emb_dim)
        self.up3 = ResBlock(base_ch, emb_dim)
        self.up2 = ResBlock(base_ch, emb_dim)
        self.up1 = ResBlock(base_ch, emb_dim)
        self.up0 = ResBlock(base_ch, emb_dim)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        if t.dim() > 1:
            print("hey, I needed to squeeze")
            t = t.squeeze(-1)
        t_emb = self.time_mlp(t.float())

        # Down
        x1 = self.in_conv(x)
        x2 = self.down1(self.pool(x1), t_emb)
        x3 = self.down2(self.pool(x2), t_emb)
        x4 = self.down3(self.pool(x3), t_emb)
        x5 = self.down4(self.pool(x4), t_emb)

        # Mid
        m = self.mid(self.pool(x5), t_emb)

        # Up
        u4 = self.up4(F.interpolate(m, scale_factor=2, mode='nearest') + x5, t_emb)
        u3 = self.up3(F.interpolate(u4, scale_factor=2, mode='nearest') + x4, t_emb)
        u2 = self.up2(F.interpolate(u3, scale_factor=2, mode='nearest') + x3, t_emb)
        u1 = self.up1(F.interpolate(u2, scale_factor=2, mode='nearest') + x2, t_emb)
        u0 = self.up0(F.interpolate(u1, scale_factor=2, mode='nearest') + x1, t_emb)
        out = self.out_conv(u0)
        return out + x
