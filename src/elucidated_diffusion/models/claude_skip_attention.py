import torch
import torch.nn as nn
import torch.nn.functional as F
from einx import rearrange, dot

# Largely from
# https://claude.ai/share/c5ca31ad-89b0-4a15-a8ee-ae2903859270
# https://claude.ai/public/artifacts/b5cc9d34-874e-4ba8-af22-72342cc9751a

# -----------------------------
# Positional embedding for timestep
# -----------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# -----------------------------
# Multi-head self-attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = rearrange('b (qkv nh dh) h w -> qkv b nh (h w) dh', 
                        qkv, qkv=3, nh=self.num_heads, dh=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = dot('b nh n dh, b nh m dh -> b nh n m', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = dot('b nh n m, b nh m dh -> b nh n dh', attn, v)
        out = rearrange('b nh (h w) dh -> b (nh dh) h w', 
                        out, nh=self.num_heads, dh=self.head_dim, h=H, w=W)
        return x + self.proj(out)


# -----------------------------
# Windowed skip-connection cross-attention for fine details
# -----------------------------
class SkipAttention(nn.Module):
    """Decoder queries encoder skip within local windows - memory efficient"""
    def __init__(self, channels, num_heads=4, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.norm_dec = nn.GroupNorm(8, channels)
        self.norm_skip = nn.GroupNorm(8, channels)
        
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Conv2d(channels, channels * 2, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, decoder_feat, skip_feat):
        B, C, H, W = decoder_feat.shape
        ws = self.window_size
        
        dec = self.norm_dec(decoder_feat)
        skip = self.norm_skip(skip_feat)
        
        q = self.to_q(dec)
        k, v = self.to_kv(skip).chunk(2, dim=1)
        
        # Reshape into windows: [B, C, H, W] -> [B, num_windows, window_size^2, C]
        q = rearrange('b (nh dh) (h ws1) (w ws2) -> b (h w) nh (ws1 ws2) dh', 
                      q, nh=self.num_heads, dh=self.head_dim, ws1=ws, ws2=ws)
        k = rearrange('b (nh dh) (h ws1) (w ws2) -> b (h w) nh (ws1 ws2) dh', 
                      k, nh=self.num_heads, dh=self.head_dim, ws1=ws, ws2=ws)
        v = rearrange('b (nh dh) (h ws1) (w ws2) -> b (h w) nh (ws1 ws2) dh', 
                      v, nh=self.num_heads, dh=self.head_dim, ws1=ws, ws2=ws)
        
        # Attention within each window: [B, num_windows, num_heads, ws^2, ws^2]
        attn = dot('b w nh n dh, b w nh m dh -> b w nh n m', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = dot('b w nh n m, b w nh m dh -> b w nh n dh', attn, v)
        out = rearrange('b (h w) nh (ws1 ws2) dh -> b (nh dh) (h ws1) (w ws2)', 
                        out, nh=self.num_heads, dh=self.head_dim, ws1=ws, ws2=ws, h=H//ws, w=W//ws)
        return decoder_feat + self.proj(out)


# -----------------------------
# Residual block with optional attention
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(emb_dim, out_ch)
        self.attn = MultiHeadAttention(out_ch) if use_attention else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.relu(self.conv1(x))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.relu(self.conv2(h))
        if self.attn is not None:
            h = self.attn(h)
        return h + self.skip(x)


# -----------------------------
# Downsampling block
# -----------------------------
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, use_attention=False):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.block = ResBlock(in_ch, out_ch, emb_dim, use_attention)
    
    def forward(self, x, t_emb):
        return self.block(self.pool(x), t_emb)


# -----------------------------
# Upsampling block with optional skip attention
# -----------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim, use_attention=False, use_skip_attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.skip_attn = SkipAttention(in_ch) if use_skip_attn else None
        self.block = ResBlock(in_ch + skip_ch, out_ch, emb_dim, use_attention)
    
    def forward(self, x, skip, t_emb):
        x_up = self.up(x)
        if self.skip_attn is not None:
            x_up = self.skip_attn(x_up, skip)
        return self.block(torch.cat([x_up, skip], dim=1), t_emb)


# -----------------------------
# Minimal 128x128 RGB U-Net with skip attention
# -----------------------------
class SkipAttentionUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

        # On 128x128 pokemon, the defaults are great.
        self.base_ch = base_ch
        self.emb_dim = emb_dim

        # Encoder
        self.inc = ResBlock(in_channels, base_ch, emb_dim)
        self.down1 = DownBlock(base_ch, base_ch*2, emb_dim)
        self.down2 = DownBlock(base_ch*2, base_ch*4, emb_dim, use_attention=True)
        self.down3 = DownBlock(base_ch*4, base_ch*4, emb_dim, use_attention=True)

        # Bottleneck
        self.mid = nn.Sequential(
            nn.AvgPool2d(2),
            ResBlock(base_ch*4, base_ch*4, emb_dim, use_attention=True)
        )

        # Decoder - skip attention at high-res stages for fine details
        self.up3 = UpBlock(base_ch*4, base_ch*4, base_ch*4, emb_dim, use_attention=True)
        self.up2 = UpBlock(base_ch*4, base_ch*4, base_ch*2, emb_dim)
        self.up1 = UpBlock(base_ch*2, base_ch*2, base_ch, emb_dim, use_skip_attn=True)
        self.up0 = UpBlock(base_ch, base_ch, base_ch, emb_dim, use_skip_attn=True)

        # Output
        self.outc = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        # Time embedding
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        t_emb = self.time_mlp(t)

        # Encoder
        x1 = self.inc(x, t_emb)            # 128x128
        x2 = self.down1(x1, t_emb)         # 64x64
        x3 = self.down2(x2, t_emb)         # 32x32
        x4 = self.down3(x3, t_emb)         # 16x16

        # Bottleneck
        m = self.mid[1](self.mid[0](x4), t_emb)  # 8x8

        # Decoder
        u3 = self.up3(m, x4, t_emb)        # 16x16
        u2 = self.up2(u3, x3, t_emb)       # 32x32
        u1 = self.up1(u2, x2, t_emb)       # 64x64 - skip attention here
        u0 = self.up0(u1, x1, t_emb)       # 128x128 - skip attention here

        return self.outc(u0)