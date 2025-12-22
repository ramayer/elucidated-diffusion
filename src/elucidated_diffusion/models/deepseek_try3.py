#  https://chat.deepseek.com/share/2d3t9k9qsq1cp7579e
#
# Extremely interesting failure modes at 128x128.
# Detailed texture / eyeballs of pokemon are excellent
# Global structure (head, legs) is good.
# But intermediate structure is chaotic.
#
# Global 
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einx import rearrange





class OptionA_FixedBottleneckUNet(nn.Module):
    """
    Option A: Fixed with correct channel dimensions
    """
    def __init__(self, in_channels=3, base_ch=64, emb_dim=128, bottleneck_size=8):
        super().__init__()
        self.base_ch = base_ch
        self.bottleneck_size = bottleneck_size
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )
        
        # =========== ENCODER ===========
        # Track exact channel counts
        self.enc1_channels = base_ch
        self.enc2_channels = base_ch * 2
        self.enc3_channels = base_ch * 4
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, self.enc1_channels, 3, padding=1),
            nn.GroupNorm(min(8, self.enc1_channels), self.enc1_channels),
            nn.SiLU(),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.enc1_channels, self.enc2_channels, 3, padding=1),
            nn.GroupNorm(min(8, self.enc2_channels), self.enc2_channels),
            nn.SiLU(),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.enc2_channels, self.enc3_channels, 3, padding=1),
            nn.GroupNorm(min(8, self.enc3_channels), self.enc3_channels),
            nn.SiLU(),
        )
        
        # =========== DOWNSAMPLING TO BOTTLENECK ===========
        # Input: enc3_channels (base_ch*4)
        # Output: enc3_channels (same, for consistency)
        self.down_to_bottleneck = nn.Sequential(
            nn.Conv2d(self.enc3_channels, self.enc3_channels, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, self.enc3_channels), self.enc3_channels),
            nn.SiLU(),
        )
        
        # =========== BOTTLENECK TRANSFORMER ===========
        self.pos_embed = nn.Parameter(
            torch.randn(1, bottleneck_size * bottleneck_size, self.enc3_channels) * 0.02
        )
        
        # Simple transformer (1 layer to start)
        self.bottleneck_transformer = TransformerBlock(self.enc3_channels, num_heads=4)
        
        # =========== UPSAMPLING FROM BOTTLENECK ===========
        # Output should match enc3_channels for decoder
        self.up_from_bottleneck = nn.Sequential(
            nn.ConvTranspose2d(self.enc3_channels, self.enc3_channels, 3, 
                              stride=2, padding=1, output_padding=1),
            nn.GroupNorm(min(8, self.enc3_channels), self.enc3_channels),
            nn.SiLU(),
        )
        
        # =========== DECODER ===========
        # Decoder 3: Processes [upsampled + enc3]
        # Input: enc3_channels * 2 (from concat)
        # Output: enc3_channels
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.enc3_channels * 2, self.enc3_channels, 3, padding=1),
            nn.GroupNorm(min(8, self.enc3_channels), self.enc3_channels),
            nn.SiLU(),
        )
        
        # Channel reduction: enc3_channels → enc2_channels
        self.reduce_channels_3_to_2 = nn.Conv2d(self.enc3_channels, self.enc2_channels, 1)
        
        # Decoder 2: Processes [reduced + enc2]
        # Input: enc2_channels * 2 (from concat)
        # Output: enc2_channels
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.enc2_channels * 2, self.enc2_channels, 3, padding=1),
            nn.GroupNorm(min(8, self.enc2_channels), self.enc2_channels),
            nn.SiLU(),
        )
        
        # Channel reduction: enc2_channels → enc1_channels
        self.reduce_channels_2_to_1 = nn.Conv2d(self.enc2_channels, self.enc1_channels, 1)
        
        # Decoder 1: Processes [reduced + enc1]
        # Input: enc1_channels * 2 (from concat)
        # Output: enc1_channels
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.enc1_channels * 2, self.enc1_channels, 3, padding=1),
            nn.GroupNorm(min(8, self.enc1_channels), self.enc1_channels),
            nn.SiLU(),
        )
        
        # =========== OUTPUT ===========
        self.out = nn.Conv2d(self.enc1_channels, in_channels, 3, padding=1)
        
        # Time conditioning
        self.time_proj1 = nn.Linear(emb_dim, self.enc1_channels)
        self.time_proj2 = nn.Linear(emb_dim, self.enc2_channels)
        self.time_proj3 = nn.Linear(emb_dim, self.enc3_channels)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.squeeze(-1).float() if t.dim() > 1 else t.float())
        
        # Time conditioning
        t1 = rearrange('b c -> b c 1 1', self.time_proj1(t_emb))
        t2 = rearrange('b c -> b c 1 1', self.time_proj2(t_emb))
        t3 = rearrange('b c -> b c 1 1', self.time_proj3(t_emb))
        
        # =========== ENCODE ===========
        x1 = self.enc1(x) + t1
        
        # Adaptive downsampling
        x2_size = (max(16, x1.shape[-2] // 2), max(16, x1.shape[-1] // 2))
        x2_input = F.interpolate(x1, size=x2_size, mode='bilinear', align_corners=False)
        x2 = self.enc2(x2_input) + t2
        
        x3_size = (max(8, x2.shape[-2] // 2), max(8, x2.shape[-1] // 2))
        x3_input = F.interpolate(x2, size=x3_size, mode='bilinear', align_corners=False)
        x3 = self.enc3(x3_input) + t3
        
        # =========== LEARNED DOWNSAMPLING ===========
        # Only downsample if x3 is larger than bottleneck_size * 2
        if x3.shape[-1] >= self.bottleneck_size * 2:
            latent = self.down_to_bottleneck(x3)
        else:
            # If already small, just pass through
            latent = x3
        
        # Ensure exact bottleneck size
        if latent.shape[-1] != self.bottleneck_size:
            latent = F.interpolate(latent, size=(self.bottleneck_size, self.bottleneck_size),
                                 mode='bilinear', align_corners=False)
        
        # =========== TRANSFORMER ===========
        B, C, H, W = latent.shape
        latent_flat = rearrange('b c h w -> b (h w) c', latent)
        
        # Add position encoding (truncate if needed)
        pos_embed = self.pos_embed[:, :latent_flat.shape[1], :]
        latent_flat = latent_flat + pos_embed
        
        # Transformer
        latent_flat = self.bottleneck_transformer(latent_flat)
        latent = rearrange('b (h w) c -> b c h w', latent_flat, h=H, w=W)
        
        # =========== LEARNED UPSAMPLING ===========
        # Upsample back to x3's resolution
        if latent.shape[-1] * 2 >= x3.shape[-1] or True:
            # Use transposed conv if it gets us close
            upsampled = self.up_from_bottleneck(latent)
            if upsampled.shape[-1] != x3.shape[-1]:
                upsampled = F.interpolate(upsampled, size=x3.shape[-2:], 
                                         mode='bilinear', align_corners=False)
        else:
            # Direct interpolation if far off
            print(f"Direct interpolation if far off {latent.shape[-1]} vs {x3.shape[-1]}")
            upsampled = F.interpolate(latent, size=x3.shape[-2:], 
                                     mode='bilinear', align_corners=False)
        
        # =========== DECODE WITH CLEAR CHANNEL MATH ===========
        # Level 3: [upsampled (C) + x3 (C)] -> C
        d3 = self.dec3(torch.cat([upsampled, x3], dim=1))
        
        # Reduce channels for next level
        d3_reduced = self.reduce_channels_3_to_2(d3)
        
        # Upsample to x2 resolution
        d2_input = F.interpolate(d3_reduced, size=x2.shape[-2:], 
                                mode='bilinear', align_corners=False)
        
        # Level 2: [d2_input (C/2) + x2 (C/2)] -> C/2
        d2 = self.dec2(torch.cat([d2_input, x2], dim=1))
        
        # Reduce channels for final level
        d2_reduced = self.reduce_channels_2_to_1(d2)
        
        # Upsample to x1 resolution
        d1_input = F.interpolate(d2_reduced, size=x1.shape[-2:], 
                                mode='bilinear', align_corners=False)
        
        # Level 1: [d1_input (C/4) + x1 (C/4)] -> C/4
        d1 = self.dec1(torch.cat([d1_input, x1], dim=1))
        
        # Final output
        return self.out(d1)

class TransformerBlock(nn.Module):
    """Simple transformer block"""
    def __init__(self, dim, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.alpha1 * attn_out
        
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.alpha2 * mlp_out
        
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# Test function with detailed debugging
def test_option_a_detailed():
    print("Testing OptionA_FixedBottleneckUNet with detailed debugging...")
    
    model = OptionA_FixedBottleneckUNet(base_ch=64, bottleneck_size=8)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test specific resolutions
    test_cases = [
        (64, 64, "64x64"),
        (128, 128, "128x128"),
        (256, 256, "256x256"),
    ]
    
    for h, w, name in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print(f"{'='*50}")
        
        x = torch.randn(2, 3, h, w)
        t = torch.randn(2)
        
        try:
            # Add debugging hooks
            debug_info = {}
            
            def save_tensor(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        debug_info[name] = {
                            'shape': output.shape,
                            'mean': output.mean().item(),
                            'std': output.std().item()
                        }
                return hook
            
            # Register hooks at key points
            hooks = []
            key_layers = [
                ('enc1', model.enc1),
                ('enc2', model.enc2),
                ('enc3', model.enc3),
                ('down_to_bottleneck', model.down_to_bottleneck),
                ('up_from_bottleneck', model.up_from_bottleneck),
                ('dec3', model.dec3),
                ('dec2', model.dec2),
                ('dec1', model.dec1),
            ]
            
            for name, layer in key_layers:
                hooks.append(layer.register_forward_hook(save_tensor(name)))
            
            # Forward pass
            output = model(x, t)
            
            # Print debug info
            print("\nLayer outputs:")
            for name, info in debug_info.items():
                print(f"  {name:20} shape: {info['shape']}")
            
            print(f"\n✓ Forward pass successful")
            print(f"  Input:  {x.shape}")
            print(f"  Output: {output.shape}")
            print(f"  Match:  {output.shape == x.shape}")
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
                
        except Exception as e:
            print(f"\n✗ Error: {e}")
            
            # Print current debug info before error
            if debug_info:
                print("\nLast successful layers:")
                for name, info in debug_info.items():
                    print(f"  {name}: {info['shape']}")
            
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n{'='*50}")
    print("Test complete!")

if __name__ == "__main__":
    test_option_a_detailed()