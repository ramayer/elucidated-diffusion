import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(emb_dim, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = F.relu(self.conv1(x))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.relu(self.conv2(h))
        return h + self.skip(x)


class GaussianRenderer(nn.Module):
    """Differentiable Gaussian splatting renderer with improved numerical stability."""
    
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, image_size)
        x_coords = torch.linspace(-1, 1, image_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('coords', torch.stack([xx, yy], dim=-1))  # [H, W, 2]
    
    def forward(self, gaussian_params):
        """
        Args:
            gaussian_params: [B, N, 8] tensor containing for each Gaussian:
                - [0:2]: position (x, y) in [-1, 1]
                - [2:4]: log scale (log_sigma_x, log_sigma_y)
                - [4:7]: color (r, g, b) in [-1, 1]
                - [7]: logit opacity
        
        Returns:
            image: [B, 3, H, W] rendered image in [-1, 1]
        """
        B, N, _ = gaussian_params.shape
        H, W = self.image_size, self.image_size
        
        # Parse Gaussian parameters
        positions = gaussian_params[:, :, 0:2]  # [B, N, 2]
        log_scales = gaussian_params[:, :, 2:4]  # [B, N, 2]
        colors = gaussian_params[:, :, 4:7]  # [B, N, 3]
        logit_opacity = gaussian_params[:, :, 7]  # [B, N]
        
        # Convert to valid ranges with clamping for stability
        log_scales = torch.clamp(log_scales, -5, 3)
        scales = torch.exp(log_scales)  # [B, N, 2]
        
        logit_opacity = torch.clamp(logit_opacity, -10, 10)
        opacity = torch.sigmoid(logit_opacity)  # [B, N]
        
        colors = torch.tanh(colors)  # [B, N, 3]
        positions = torch.clamp(positions, -2, 2)
        
        # Expand coordinates for batch processing
        coords = self.coords.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
        positions = positions.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 2]
        scales = scales.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 2]
        
        # Compute squared distances
        diff = coords - positions  # [B, N, H, W, 2]
        scaled_diff = diff / (scales + 1e-4)
        dist_sq = (scaled_diff ** 2).sum(dim=-1)  # [B, N, H, W]
        dist_sq = torch.clamp(dist_sq, 0, 50)
        
        # Gaussian weights
        weights = torch.exp(-0.5 * dist_sq)  # [B, N, H, W]
        weights = weights * opacity.unsqueeze(2).unsqueeze(2)  # [B, N, H, W]
        
        # Alpha compositing
        total_weight = weights.sum(dim=1, keepdim=True) + 1e-5  # [B, 1, H, W]
        weights_normalized = weights / total_weight  # [B, N, H, W]
        
        # Blend colors
        colors = colors.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 3]
        image = (weights_normalized.unsqueeze(-1) * colors).sum(dim=1)  # [B, H, W, 3]
        
        # Add background (neutral gray at 0.0)
        background_weight = 1.0 - total_weight.squeeze(1)  # [B, H, W]
        image = image + background_weight.unsqueeze(-1) * 0.0
        
        # Clamp and convert to [B, C, H, W]
        image = torch.clamp(image, -1, 1)
        image = image.permute(0, 3, 1, 2)
        
        return image


class GaussianSplattingDiffusionUNet(nn.Module):
    """
    U-Net that predicts Gaussian splats from bottleneck, then refines with
    a lightweight residual that takes the rendered Gaussians + skip connections.
    """
    
    def __init__(self, in_channels=3, base_ch=64, emb_dim=128, image_size=128, num_gaussians=None):
        super().__init__()
        
        self.image_size = image_size
        
        # Auto-determine number of Gaussians based on image size if not specified
        if num_gaussians is None:
            num_gaussians = (image_size // 8) ** 2
            num_gaussians = min(num_gaussians, 512)
            num_gaussians = max(num_gaussians, 32)
        
        self.num_gaussians = num_gaussians
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.inc = ResBlock(in_channels, base_ch, emb_dim)
        self.down1 = ResBlock(base_ch, base_ch*2, emb_dim)
        self.down2 = ResBlock(base_ch*2, base_ch*4, emb_dim)
        self.down3 = ResBlock(base_ch*4, base_ch*4, emb_dim)
        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.mid = ResBlock(base_ch*4, base_ch*4, emb_dim)

        # Predict Gaussians from bottleneck + time embedding
        self.gaussian_pooling = nn.AdaptiveAvgPool2d((4, 4))
        gaussian_input_dim = base_ch * 4 * 4 * 4
        
        # Add time embedding to Gaussian predictor
        self.gaussian_time_mlp = nn.Linear(emb_dim, 256)
        
        self.gaussian_mlp = nn.Sequential(
            nn.Linear(gaussian_input_dim + 256, 512),  # +256 for time embedding
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.renderer = GaussianRenderer(image_size)
        
        # Initialize Gaussian head with small weights
        self.gaussian_head = nn.Linear(512, num_gaussians * 8)
        nn.init.normal_(self.gaussian_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gaussian_head.bias)
        
        # Lightweight residual refinement network
        # Takes: rendered Gaussians + skip connection from early encoder + time embedding
        # The residual will learn how much to denoise based on noise level
        
        # Time embedding for residual
        self.residual_time_mlp = nn.Sequential(
            nn.Linear(emb_dim, base_ch),
            nn.ReLU()
        )
        
        # Learnable denoising schedule - predicts how much to trust Gaussians vs keep noise
        self.denoise_schedule = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]: 0=keep all noise, 1=fully trust Gaussians
        )
        
        self.residual_refine = nn.Sequential(
            # Input: 3 (rendered) + base_ch (x1 skip)
            nn.Conv2d(in_channels + base_ch, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch // 2, in_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x, t):
        B, C, H, W = x.shape
        
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        t_emb = self.time_mlp(t)

        # Encoder - save skip connections
        x1 = self.inc(x, t_emb)  # Save this for residual
        x2 = self.down1(self.pool(x1), t_emb)
        x3 = self.down2(self.pool(x2), t_emb)
        x4 = self.down3(self.pool(x3), t_emb)

        # Bottleneck
        m = self.mid(self.pool(x4), t_emb)

        # Predict Gaussians from bottleneck WITH time embedding
        pooled = self.gaussian_pooling(m)  # [B, base_ch*4, 4, 4]
        flat = pooled.flatten(start_dim=1)
        
        # Add time information to Gaussian predictor
        time_features = self.gaussian_time_mlp(t_emb)  # [B, 256]
        gaussian_input = torch.cat([flat, time_features], dim=1)
        
        gaussian_features = self.gaussian_mlp(gaussian_input)  # [B, 512]
        gaussian_params = self.gaussian_head(gaussian_features)  # [B, N*8]
        gaussian_params = gaussian_params.view(B, self.num_gaussians, 8)
        
        # Render Gaussians (coarse structure)
        rendered_gaussians = self.renderer(gaussian_params)
        
        # Learn how much to denoise based on time
        # High t (noisy) → low denoise_strength (keep more noise)
        # Low t (clean) → high denoise_strength (trust Gaussians more)
        denoise_strength = self.denoise_schedule(t_emb)  # [B, 1]
        denoise_strength = denoise_strength.view(B, 1, 1, 1)
        
        # Blend noisy input with Gaussian rendering
        # This preserves high-frequency noise structure while adding Gaussian content
        denoised_base = x * (1 - denoise_strength) + rendered_gaussians * denoise_strength
        
        # Residual refinement sees: blended result + early skip features
        residual_input = torch.cat([denoised_base, x1], dim=1)
        
        # First conv layer, then add time embedding
        residual_features = self.residual_refine[0](residual_input)  # First conv
        residual_features = self.residual_refine[1](residual_features)  # ReLU
        
        # Inject time embedding
        time_spatial = self.residual_time_mlp(t_emb)[:, :, None, None]
        residual_features = residual_features + time_spatial
        
        # Continue through rest of network
        for layer in self.residual_refine[2:]:
            residual_features = layer(residual_features)
        
        residual_correction = residual_features
        
        # Apply small residual correction
        output = denoised_base + 0.2 * residual_correction
        output = torch.clamp(output, -1, 1)
        
        return output


# Convenience alias matching baseline naming
ChatGPTDiffusionUNet128 = GaussianSplattingDiffusionUNet


# Test the model
if __name__ == "__main__":
    # Test at 64x64
    model_64 = GaussianSplattingDiffusionUNet(base_ch=64, image_size=64)
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    out = model_64(x, t)
    print(f"64x64 test - Input: {x.shape}, Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"Number of Gaussians: {model_64.num_gaussians}")
    print(f"Parameters: {sum(p.numel() for p in model_64.parameters()) / 1e6:.2f}M")
    
    # Test at 128x128
    model_128 = GaussianSplattingDiffusionUNet(base_ch=64, image_size=128)
    x = torch.randn(2, 3, 128, 128)
    t = torch.randint(0, 1000, (2,))
    out = model_128(x, t)
    print(f"\n128x128 test - Input: {x.shape}, Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"Number of Gaussians: {model_128.num_gaussians}")
    print(f"Parameters: {sum(p.numel() for p in model_128.parameters()) / 1e6:.2f}M")
    
    # Test gradient stability
    print("\nTesting gradient stability...")
    model_64.train()
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    t = torch.randint(0, 1000, (2,))
    out = model_64(x, t)
    loss = out.mean()
    loss.backward()
    print(f"Gradients computed successfully. Max grad: {x.grad.abs().max():.6f}")