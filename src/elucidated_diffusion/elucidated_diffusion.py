import torch
import torch.nn.functional as F

# EDM parameters (keeping your existing values)
P_mean = -1.2
P_std = 1.2
sigma_data = 0.5
sigma_min = 0.002
sigma_max = 80
rho = 7

def edm_sigma_schedule(t):
    return (sigma_max ** (1/rho) + t * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho

def edm_loss_weight(sigma, sigma_data=sigma_data):
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

def unified_edm_sampling(model, num_steps=18, batch_size=8, img_shape=(3, 256, 256), 
                        lr_conditioning=None, headstart_sigma=None):
    """
    Unified EDM ancestral sampling for both diffusion and super-resolution.
    
    Args:
        model: The diffusion model (UNet or HybridViTUNetSR)
        num_steps: Number of sampling steps
        batch_size: Batch size
        img_shape: Output image shape (C, H, W)
        lr_conditioning: Optional LR images for SR [B, 3, 64, 64]. If None, pure generation.
        headstart_sigma: Optional noise level for headstart (SR only)
        
    Returns:
        Generated images [B, C, H, W]
    """
    device = next(model.parameters()).device
    is_super_resolution = lr_conditioning is not None
    
    # Initialize noise
    if headstart_sigma is not None and is_super_resolution:
        # Head-start from upsampled LR + noise
        lr_up = F.interpolate(lr_conditioning, size=img_shape[-2:], mode='bilinear', align_corners=False)
        x_next = lr_up + headstart_sigma * torch.randn((batch_size,) + img_shape, device=device)
    else:
        # Pure Gaussian noise initialization
        x_next = torch.randn((batch_size,) + img_shape, device=device)
    
    # Time step schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    
    if headstart_sigma is not None and is_super_resolution:
        # Adjusted schedule for headstart
        t_steps = (headstart_sigma ** (1/rho) + step_indices / (num_steps - 1) * 
                  (sigma_min ** (1/rho) - headstart_sigma ** (1/rho))) ** rho
    else:
        # Standard schedule
        t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * 
                  (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        # Scale initial noise if no headstart
        x_next = x_next * t_steps[0]
    
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    
    # Main sampling loop (Heun's method)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Preconditioning coefficients for current timestep
        sigma = t_cur.float()
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        # Model forward pass - handle both diffusion and SR cases
        if is_super_resolution:
            # Super-resolution: pass noisy HR and LR conditioning separately
            F_x = model(c_in * x_cur, lr_conditioning, c_noise.expand(batch_size))
        else:
            # Pure diffusion: only noisy image, no conditioning
            F_x = model(c_in * x_cur, c_noise.expand(batch_size))
        
        # Euler step
        denoised = c_skip * x_cur + c_out * F_x
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        
        # Apply 2nd order correction (Heun's method) - except for last step
        if i < num_steps - 1:
            # Preconditioning coefficients for next timestep
            sigma_next = t_next.float()
            c_skip_next = sigma_data ** 2 / (sigma_next ** 2 + sigma_data ** 2)
            c_out_next = sigma_next * sigma_data / (sigma_next ** 2 + sigma_data ** 2).sqrt()
            c_in_next = 1 / (sigma_data ** 2 + sigma_next ** 2).sqrt()
            c_noise_next = sigma_next.log() / 4
            
            # Second model call for Heun's method
            if is_super_resolution:
                F_x_next = model(c_in_next * x_next, lr_conditioning, c_noise_next.expand(batch_size))
            else:
                F_x_next = model(c_in_next * x_next, c_noise_next.expand(batch_size))
            
            denoised_next = c_skip_next * x_next + c_out_next * F_x_next
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    
    return x_next.to("cpu")

# Convenience wrapper functions for cleaner API
def edm_generate_from_noise(model, num_steps=18, batch_size=8, img_shape=(3, 64, 64)):
    """
    Generate images from pure noise (unconditional generation).
    
    Args:
        model: Diffusion model
        num_steps: Sampling steps  
        batch_size: Batch size
        img_shape: Output shape (C, H, W)
    """
    return unified_edm_sampling(
        model=model,
        num_steps=num_steps, 
        batch_size=batch_size,
        img_shape=img_shape,
        lr_conditioning=None,  # No conditioning = pure generation
        headstart_sigma=None
    )

def edm_super_resolve(model, lr_images, num_steps=18, headstart_sigma=None):
    """
    Super-resolve LR images to HR.
    
    Args:
        model: Super-resolution model (e.g., HybridViTUNetSR)
        lr_images: LR conditioning images [B, 3, 64, 64]
        num_steps: Sampling steps
        headstart_sigma: Optional noise level for headstart
    """
    batch_size = lr_images.shape[0]
    img_shape = (3, 256, 256)  # HR output shape
    
    return unified_edm_sampling(
        model=model,
        num_steps=num_steps,
        batch_size=batch_size, 
        img_shape=img_shape,
        lr_conditioning=lr_images,
        headstart_sigma=headstart_sigma
    )

# Backward compatibility - keep your existing function names
def edm_ancestral_sampling_for_diffusion(model, num_steps=18, batch_size=8, img_shape=(3, 64, 64)):
    """Backward compatibility wrapper"""
    return edm_generate_from_noise(model, num_steps, batch_size, img_shape)

def edm_ancestral_sampling_for_sr(model, lr_64, num_steps=18, batch_size=8, img_shape=(3, 256, 256), 
                                 headstart_sigma=None):
    """Backward compatibility wrapper"""
    return edm_super_resolve(model, lr_64, num_steps, headstart_sigma)


# Example usage:
if __name__ == "__main__":
    # Mock model for testing
    class MockModel:
        def __call__(self, *args):
            if len(args) == 2:  # Diffusion case
                return torch.randn_like(args[0])
            else:  # SR case  
                return torch.randn_like(args[0])
        def parameters(self):
            return [torch.tensor(0.0)]
    
    model = MockModel()
    
    # Test pure generation
    generated = edm_generate_from_noise(model, batch_size=2, img_shape=(3, 64, 64))
    print(f"Generated shape: {generated.shape}")
    
    # Test super-resolution
    lr_batch = torch.randn(2, 3, 64, 64)
    sr_result = edm_super_resolve(model, lr_batch, headstart_sigma=5.0)
    print(f"Super-resolved shape: {sr_result.shape}")