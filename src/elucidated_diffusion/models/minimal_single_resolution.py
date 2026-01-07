import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Time embedding ---

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        x: (batch,) or (batch, 1) with continuous time
        returns: (batch, dim)
        """
        if x.dim() > 1:
            x = x.squeeze(-1)
        device = x.device
        half_dim = self.dim // 2
        emb_scale = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb_freq = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        # (batch, half_dim)
        emb = x[:, None] * emb_freq[None, :]
        # (batch, dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# --- Single-resolution ResNet block ---

class ResBlock(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.time_mlp = nn.Linear(emb_dim, ch)

    def forward(self, x, t_emb):
        """
        x: (batch, ch, H, W)
        t_emb: (batch, emb_dim)
        """
        h = self.conv(x)
        # broadcast time embedding to spatial dims
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.relu(h)
        return x + h  # residual


# --- Extreme minimal EDM model ---

class MinimalSingleResEDM(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, depth=8, emb_dim=128):
        """
        in_channels: image channels
        base_ch: feature channels (kept constant across all blocks)
        depth: number of ResBlocks
        emb_dim: time embedding dimension
        """
        super().__init__()

        # time embedding: sinusoidal + small MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

        # input / output projections
        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 3, padding=1)

        # stack of same-resolution ResBlocks
        self.blocks = nn.ModuleList(
            [ResBlock(base_ch, emb_dim) for _ in range(depth)]
        )

    def forward(self, x, t):
        """
        x: (batch, in_channels, H, W)
        t: (batch,) or (batch, 1)
        returns: (batch, in_channels, H, W)
        """
        if t.dim() > 1:
            t = t.squeeze(-1)
        t = t.float()
        t_emb = self.time_mlp(t)

        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_conv(h)
