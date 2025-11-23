# First try: https://claude.ai/public/artifacts/02b7dc89-f138-4f89-8f00-594e84b95977
# Second try: https://claude.ai/public/artifacts/12fa7371-6e13-45c7-abe4-c2cfbaa2f1f7
# Third Try https://claude.ai/public/artifacts/6579f627-a0a2-45fc-beb1-f147ac09c59e

"""## Novel Aspects of the Gradual Conv→Transformer Transition Architecture

### Related Work in Hybrid Conv-Transformer Architectures

**Existing Approaches:**


1. **[Systematic Review of Hybrid Vision Transformer Architectures for Radiological Image Analysis](https://www.medrxiv.org/content/10.1101/2024.06.21.24309265v1.full) - "Vision Transformer (ViT) and Convolutional Neural Networks (CNNs) each possess distinct strengths in medical imaging: ViT excels in capturing long-range dependencies through self-attention, while CNNs are adept at extracting local features via spatial convolution filters." - but focuses on leveraging complementary strengths rather than progressive transition.

2. **[CvT (2021)](https://arxiv.org/abs/2103.15808)**: Introduces convolutions into ViT through ["a hierarchy of Transformers containing a new convolutional token embedding, and a convolutional Transformer block"](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf) - but this adds conv operations within transformer blocks, not a gradual transition.

3. **[Swin Transformer (2021)](https://arxiv.org/abs/2103.14030)**: ["Hierarchical Vision Transformer using Shifted Windows"](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) - uses hierarchical processing but remains pure transformer.

4. **Hybrid CNN-transformer networks**: "Convolutional Neural Networks (CNN) are highly effective at capturing local details, whereas Transformers is effective for modeling long-range dependencies" - but these typically use parallel or sequential processing, not gradual transition.

### What Makes Our "Gradual Transition" Novel

#### **1. Progressive Attention Weighting**
Most hybrid architectures use **binary decisions** - a layer is either conv OR transformer. Our approach uses **weighted blending**:
```
Layer 1: 100% conv, 0% attention
Layer 2: 75% conv, 25% attention  
Layer 3: 25% conv, 75% attention
Layer 4: 0% conv, 100% attention
```

This **progressive transition** hasn't been explored in the literature we surveyed.

#### **2. Depth-Dependent Architecture Philosophy**
Our approach embodies a specific computational philosophy:
- **Early layers**: Pure spatial locality (conv) for fine details
- **Middle layers**: Gradual introduction of global context (hybrid)
- **Deep layers**: Pure global reasoning (transformer)

This mirrors how **biological visual processing** works (V1 → V2 → V4 → IT cortex) but we didn't find computer vision architectures that explicitly implement this progression.

#### **3. Resolution-Aware Hybrid Design**
The transition aligns with resolution:
- **64×64 → 32×32**: Spatial details matter most → pure conv
- **16×16 → 8×8**: Global structure emerges → heavy attention  
- **4×4**: Pure global reasoning → pure transformer

#### **4. Training Dynamics Implications**
Our approach should produce unique **training dynamics**:
- **Early training**: Conv layers learn quickly (local patterns)
- **Mid training**: Attention weights gradually optimize
- **Late training**: Global coherence emerges

This **progressive learning curriculum** is built into the architecture itself, rather than being externally imposed.

### Novel Aspects Summary

**Architecture-level novelty:**
- **Progressive attention weighting** (not binary conv/transformer choice)
- **Depth-dependent transition philosophy** (matching biological vision)
- **Resolution-aligned hybrid strategy**

**Training-level novelty:**
- **Built-in curriculum learning** (fast local → slow global)
- **Gradual capacity scaling** (simple → complex representations)

**Diffusion-specific novelty:**
- **Scale-appropriate denoising**: Conv for high-freq noise, transformers for structure
- **Progressive global understanding**: Matches diffusion's coarse→fine generation process

Our approach appears genuinely novel in the **progressive weighting** and **resolution-aware transition** aspects. The training dynamics observed (structured blob→limb development) might be evidence that this architectural philosophy is working as intended.

### References

- Wu, H., Xiao, B., Codella, N., Liu, M., Dai, X., Yuan, L., & Zhang, L. (2021). [CvT: Introducing convolutions to vision transformers](https://arxiv.org/abs/2103.15808). *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 22-28.
- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). [Swin transformer: Hierarchical vision transformer using shifted windows](https://arxiv.org/abs/2103.14030). *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 10012-10022.
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Reuse your existing components
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)) 
        )
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# class EfficientAttention(nn.Module):
#     """Memory-efficient attention with fewer parameters"""
#     def __init__(self, channels, num_heads=4, qkv_bias=False):
#         super().__init__()
#         assert channels % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = channels // num_heads
#         self.scale = self.head_dim ** -0.5
        
#         # Use GroupNorm for better memory efficiency than LayerNorm
#         self.norm = nn.GroupNorm(8, channels)
#         # Combine qkv into single conv for efficiency
#         self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=qkv_bias)
#         self.proj = nn.Conv2d(channels, channels, kernel_size=1)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         h = self.norm(x)
        
#         # Memory-efficient attention computation
#         qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H*W).permute(1, 0, 2, 4, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]
        
#         # Use scaled dot-product attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = F.softmax(attn, dim=-1)
        
#         out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
#         return x + self.proj(out)

class EfficientAttention(nn.Module):
    """
    Memory-efficient attention with lightweight MLP.
    
    Implements a complete transformer block with both attention and MLP components:
    1. Multi-head self-attention: Mixes information across spatial locations (linear operation)
    2. MLP with GELU: Applies non-linear per-location feature transformation (essential!)
    
    Why the MLP is critical:
    - Attention alone is a linear operation (weighted averaging)
    - Without MLP, stacking attention layers collapses to single linear transform
    - MLP provides the non-linearity needed for complex feature learning
    - Typical transformers use mlp_ratio=4.0; we use 2.0 for memory efficiency
    
    Architecture pattern (standard transformer):
    - x = x + attention(norm1(x))   # First residual: spatial mixing
    - x = x + mlp(norm2(x))         # Second residual: feature transformation
    
    Memory trade-off:
    - mlp_ratio=2.0 adds ~30% parameters vs attention-only
    - Still much cheaper than mlp_ratio=4.0 (standard ViT)
    - Essential for transformer expressiveness and capacity
    """
    def __init__(self, channels, num_heads=4, mlp_ratio=2.0, qkv_bias=False):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Attention components
        self.norm1 = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # MLP components (essential for transformer expressiveness!)
        self.norm2 = nn.GroupNorm(8, channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, channels, 1)
        )

    def forward(self, x):
        # Multi-head self-attention block
        B, C, H, W = x.shape
        h = self.norm1(x)
        
        # Attention computation
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H*W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        attn_out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        x = x + self.proj(attn_out)
        
        # MLP block (critical for non-linear feature transformation!)
        x = x + self.mlp(self.norm2(x))
        
        return x
    
class ConvBlock(nn.Module):
    """Pure convolutional block - no attention"""
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.gelu(self.conv1(x))  # GELU generally better than ReLU for transformers
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.gelu(self.conv2(h))
        return h + self.skip(x)

class HybridBlock(nn.Module):
    """Hybrid conv + lightweight attention block"""
    def __init__(self, in_ch, out_ch, emb_dim, num_heads=4, attention_ratio=0.5):
        super().__init__()
        self.conv_block = ConvBlock(in_ch, out_ch, emb_dim)
        self.attention = EfficientAttention(out_ch, num_heads)
        self.attention_ratio = attention_ratio  # How much attention vs conv
        
    def forward(self, x, t_emb):
        # Conv processing
        h_conv = self.conv_block(x, t_emb)
        # Attention processing
        h_attn = self.attention(h_conv)
        # Weighted combination (gradually favor attention in deeper layers)
        return h_conv * (1 - self.attention_ratio) + h_attn * self.attention_ratio

class ViTBlock(nn.Module):
    """Lightweight ViT-style block for deeper layers"""
    def __init__(self, in_channels, out_channels, emb_dim, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        # Add channel projection if input/output channels differ
        self.channel_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.time_mlp = nn.Linear(emb_dim, out_channels)
        
        # Attention (operates on out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.attn = EfficientAttention(out_channels, num_heads)
        
        # MLP (reduced ratio for memory efficiency)
        self.norm2 = nn.GroupNorm(8, out_channels)
        mlp_hidden = int(out_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, out_channels, 1)
        )
        
    def forward(self, x, t_emb):
        # Channel projection first
        x = self.channel_proj(x)
        # Time conditioning
        x = x + self.time_mlp(t_emb)[:, :, None, None]
        # Attention
        x = x + self.attn(x)
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class GradualTransitionUNet(nn.Module):
    """
    Progressive conv→ViT transition architecture:
    
    64x64 → 32x32: Pure conv (fine spatial details)
    32x32 → 16x16: Light hybrid (25% attention, 75% conv)  
    16x16 → 8x8:  Heavy hybrid (75% attention, 25% conv)
    8x8 → 4x4:    Pure ViT (global understanding)
    
    Memory optimizations:
    - Reduced head counts
    - Smaller MLP ratios
    - Efficient attention implementation
    - No unnecessary cross-attention modules
    """
    def __init__(self, in_channels=3, base_ch=64, emb_dim=128):
        super().__init__()
        
        # Time embedding (same as yours)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )
        
        # Encoder: Gradual transition from conv to ViT
        # Stage 1: Pure conv (64x64 → 32x32) - fine details need spatial locality
        self.enc1 = ConvBlock(in_channels, base_ch, emb_dim)
        
        # Stage 2: Light hybrid (32x32 → 16x16) - start introducing attention
        self.enc2 = HybridBlock(base_ch, base_ch*2, emb_dim, num_heads=4, attention_ratio=0.25)
        
        # Stage 3: Heavy hybrid (16x16 → 8x8) - more attention for mid-level features  
        self.enc3 = HybridBlock(base_ch*2, base_ch*4, emb_dim, num_heads=4, attention_ratio=0.75)
        
        # Stage 4: Pure ViT (8x8 → 4x4) - global understanding at low resolution
        self.enc4 = ViTBlock(base_ch*4, base_ch*4, emb_dim, num_heads=8, mlp_ratio=2.0)
        
        # Bottleneck: Pure ViT processing (4x4)
        self.bottleneck1 = ViTBlock(base_ch*4, base_ch*4, emb_dim, num_heads=8, mlp_ratio=2.0)
        self.bottleneck2 = ViTBlock(base_ch*4, base_ch*4, emb_dim, num_heads=8, mlp_ratio=2.0)
        
        # Decoder: Reverse transition ViT → conv
        # Fixed channel calculations for proper concatenation
        self.dec4 = ViTBlock(base_ch*4*2, base_ch*4, emb_dim, num_heads=8, mlp_ratio=2.0)        # 512 in → 256 out
        self.dec3 = HybridBlock(base_ch*4 + base_ch*4, base_ch*2, emb_dim, num_heads=4, attention_ratio=0.75)  # 256+256=512 in → 128 out
        self.dec2 = HybridBlock(base_ch*2 + base_ch*2, base_ch, emb_dim, num_heads=4, attention_ratio=0.25)    # 128+128=256 in → 64 out
        self.dec1 = ConvBlock(base_ch + base_ch, base_ch, emb_dim)                            # 64+64=128 in → 64 out
        
        # Standard UNet operations
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Output
        self.outc = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        """Same signature as your existing models"""
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        t_emb = self.time_mlp(t)

        # Progressive conv→ViT encoder
        e1 = self.enc1(x, t_emb)                    # [B, 64, 64, 64] - Pure conv
        e2 = self.enc2(self.pool(e1), t_emb)        # [B, 128, 32, 32] - Light hybrid (25% attention)
        e3 = self.enc3(self.pool(e2), t_emb)        # [B, 256, 16, 16] - Heavy hybrid (75% attention)
        e4 = self.enc4(self.pool(e3), t_emb)        # [B, 256, 8, 8] - Pure ViT

        # Pure ViT bottleneck
        m = self.pool(e4)                           # [B, 256, 4, 4]
        m = self.bottleneck1(m, t_emb)              # Pure ViT processing
        m = self.bottleneck2(m, t_emb)              # Pure ViT processing

        # Progressive ViT→conv decoder  
        d4 = self.dec4(torch.cat([self.up(m), e4], dim=1), t_emb)      # Pure ViT
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1), t_emb)     # Heavy hybrid
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)     # Light hybrid  
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)     # Pure conv

        return self.outc(d1)

def create_gradual_transition_unet(config='efficient'):
    """
    Factory for memory-efficient configurations
    
    'efficient': Optimized for 6GB GPU
    'balanced': More capacity, needs ~8GB
    'minimal': Very lightweight for 4GB GPU

    This model at Balanced can draw beautiful new pokemon in 12 hours.
    """
    configs = {
        'minimal': {
            'base_ch': 48,  # Smaller base channels
            'emb_dim': 96   # Smaller embedding
        },
        'efficient': {
            'base_ch': 64,  # Your current size
            'emb_dim': 128  # Your current embedding
        },
        'balanced': {
            'base_ch': 80,  # Slightly larger
            'emb_dim': 160  # Larger embedding
        }
    }
    
    return GradualTransitionUNet(**configs[config])

# Memory estimation helper
def estimate_memory_usage(model, batch_size=4, image_size=64):
    """Rough memory estimation"""
    model_params = sum(p.numel() for p in model.parameters())
    # Rough estimate: 4 bytes per param + 4x for gradients + activation memory
    model_memory = model_params * 4 * 5  # Model + gradients + optimizer states
    activation_memory = batch_size * 3 * image_size * image_size * 4 * 10  # Rough activation estimate
    total_gb = (model_memory + activation_memory) / (1024**3)
    return {
        'model_params': f"{model_params:,}",
        'estimated_gb': f"{total_gb:.2f}GB"
    }

# Usage example:
if __name__ == "__main__":
    model = create_gradual_transition_unet('efficient')
    print("Gradual Transition UNet created!")
    
    # Memory check
    mem_info = estimate_memory_usage(model)
    print(f"Parameters: {mem_info['model_params']}")
    print(f"Estimated memory: {mem_info['estimated_gb']}")
    
    # Quick test
    x = torch.randn(2, 3, 64, 64)
    t = torch.randn(2)
    
    with torch.no_grad():
        out = model(x, t)
    print(f"Output shape: {out.shape}")
    print("Forward pass successful!")
