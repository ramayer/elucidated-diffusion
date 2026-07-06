import copy
import torch
import math
class EMAHelper:
    """
    EMAHelper implements Exponential Moving Average (EMA) tracking for PyTorch models.
    This helper class maintains a frozen copy of a given model and updates its parameters
    using an exponential moving average of the original model's parameters. The EMA decay
    rate can be dynamically adjusted based on the training step, allowing for smooth
    transition from a lower to a higher decay value.
    EMAHelper is generic and can be used with any PyTorch model, including both Diffusion Models
    and SuperResolution Models within this project. It is designed to improve model stability
    and performance during training by providing a smoothed version of the model for evaluation
    or inference.
    Args:
        model (torch.nn.Module): The model to track with EMA.
        d_final (float, optional): The final EMA decay rate. Default is 0.999.
    Attributes:
        ema_model (torch.nn.Module): The EMA-tracked copy of the model (in eval mode).
        step (int): The current training step for dynamic decay calculation.
    Methods:
        get_ema_decay(): Computes the current EMA decay rate.
        update(model): Updates the EMA model parameters using the current model.
        get_model(): Returns the EMA-smoothed model (in eval mode).
    """

    def __init__(self, model, step=1, d_final=0.999):
        # Make a frozen copy of the model for EMA tracking
        self.ema_model = copy.deepcopy(model).eval()
        self.step = step
        self.d_final = d_final
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def get_ema_decay(self, step = None):
        """
        Progressive EMA decay that starts loose and tightens,
        but always slightly favors newer weights.
        """
        step = step or self.step
        # Start with short effective window ~ step
        d = 1 - 1/(step+1)
        # Gradually push it closer to base as step grows
        return d * self.d_final

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        decay = self.get_ema_decay()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                v.copy_(decay * v + (1 - decay) * msd[k])
        self.step += 1

    def get_model(self):
        """Return the EMA-smoothed model (already in eval mode)."""
        return self.ema_model

"""
Gemini's review:

Yes, your implementation is structurally sound, but it contains one critical 
performance bug and a few logic issues that will break or heavily slow down 
your Elucidated Diffusion training.Here is the breakdown of what is wrong 
and how to fix it.

1. The Critical Bug: state_dict() in the Training Loop

Inside your update method, you call model.state_dict() 
and self.ema_model.state_dict().

The Problem: In PyTorch, .state_dict() constructs a brand-new Python 
dictionary and copies references every single time it is called. Doing 
this every training step creates massive CPU overhead and completely 
breaks GPU asynchronous execution (CUDA graphs / stream synchronization).

The Fix: Iterate directly over model.all_modules() or model.parameters() and model.buffers().

2. The Logic Flaw: Missing Buffers

Diffusion models rely heavily on running statistics stored in 
buffers (like BatchNormalization or tracking metrics). Your current loop only 
updates weights if you only look at parameters, but copying state_dict() keys 
like you did can mix up tensors if not handled carefully. More importantly, 
your formula handles parameters well but completely freezes buffers or 
updates them with decay, whereas buffers should typically be copied 
directly or handled separately.

3. The Math Issue: Decay Warmup is Too Short

Your warmup formula d = 1 - 1/(step+1) ramps up incredibly fast.

At step 10, your decay multiplier is already 0.909.At step 1000, 
it is 0.999.For a LAION-5B training run where you will do hundreds 
of thousands of steps, this warmup disappears in the first few 
minutes of training, making it effectively useless. It does not 
match the long half-life ramp-up recommended by NVIDIA EDM.
"""

import copy
import math
import torch

class EDMEMAHelper:
    """
    ```
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_helper.ema_model.state_dict(),  # Save the EMA weights!
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, 'checkpoint.pt')
    ```
    and
    ```
        checkpoint = torch.load('checkpoint.pt')

        # 1. Load the main model
        model.load_state_dict(checkpoint['model_state_dict'])

        # 2. Reinitialize the helper with the exact step you left off on
        ema_helper = EMAHelper(model, step=checkpoint['step'], batch_size=512)

        # 3. OVERWRITE the freshly copied weights with the saved EMA weights
        ema_helper.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

    ```
    """
    def __init__(self, model, step=1, batch_size=512, ema_halflife_kimg=500):
        """
        Optimized EMA Helper tailored for NVIDIA Elucidated Diffusion (EDM).
        """
        # Create a deep copy and freeze it
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
            
        self.step = step
        self.batch_size = batch_size
        self.ema_halflife_kimg = ema_halflife_kimg
        
        # Pre-cache parameter lists to avoid state_dict() overhead in the loop
        self.ema_params = list(self.ema_model.parameters())
        self.model_params = list(model.parameters())
        
        self.ema_buffers = list(self.ema_model.buffers())
        self.model_buffers = list(model.buffers())

    def get_ema_decay(self):
        """
        NVIDIA EDM profile: Computes step-wise decay based on image half-life.
        Adapts perfectly to your batch size.
        """
        # Current total images processed (in thousands)
        current_kimg = (self.step * self.batch_size) / 1000.0
        
        # Ramp up the halflife target based on progress
        # This replaces your (1 - 1/step) with a scale matching millions of images
        current_halflife = min(self.ema_halflife_kimg, current_kimg * 2) 
        
        # Calculate the exact step-wise decay rate
        decay = math.exp(math.log(0.5) / (current_halflife * 1000 / self.batch_size))
        return decay

    @torch.no_grad()
    def update(self, model):
        decay = self.get_ema_decay()

        # 1. Update Parameters (Smooth Moving Average)
        for ema_p, model_p in zip(self.ema_params, self.model_params):
            ema_p.mul_(decay).add_(model_p, alpha=1 - decay)
            
        # 2. Update Buffers (Direct copy for tracking stats like BatchNorm)
        for ema_b, model_b in zip(self.ema_buffers, self.model_buffers):
            ema_b.copy_(model_b)
            
        self.step += 1

    def get_model(self):
        return self.ema_model

import copy
import math
import torch

class CPUOffloadedEMAHelper:
    def __init__(self, model, step=1, batch_size=512, ema_halflife_kimg=500):
        """
        EMA Helper that stores the EMA weights completely on the CPU,
        freeing up maximum GPU VRAM for training.
        """
        self.step = step
        self.batch_size = batch_size
        self.ema_halflife_kimg = ema_halflife_kimg
        
        # 1. Create the EMA model clone directly on the CPU
        self.ema_model = copy.deepcopy(model).cpu().eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
            
        # 2. Pin the CPU memory so the GPU can stream data to it instantly
        for p in self.ema_model.parameters():
            p.pin_memory()
        for b in self.ema_model.buffers():
            b.pin_memory()

        # Cache parameter references
        self.ema_params = list(self.ema_model.parameters())
        self.ema_buffers = list(self.ema_model.buffers())

    def get_ema_decay(self):
        current_kimg = (self.step * self.batch_size) / 1000.0
        current_halflife = min(self.ema_halflife_kimg, current_kimg * 2) 
        return math.exp(math.log(0.5) / (current_halflife * 1000 / self.batch_size))

    @torch.no_grad()
    def update(self, model):
        decay = self.get_ema_decay()

        # Iterate through the active GPU model parameters
        for ema_p, model_p in zip(self.ema_params, model.parameters()):
            # non_blocking=True streams the GPU tensor to the CPU asynchronously
            gpu_p_copied_to_cpu = model_p.to('cpu', non_blocking=True)
            
            # Do the EMA math strictly on the CPU
            ema_p.mul_(decay).add_(gpu_p_copied_to_cpu, alpha=1 - decay)
            
        # Do the same async streaming for tracking buffers (BatchNorm, etc.)
        for ema_b, model_b in zip(self.ema_buffers, model.buffers()):
            gpu_b_copied_to_cpu = model_b.to('cpu', non_blocking=True)
            ema_b.copy_(gpu_b_copied_to_cpu)
            
        self.step += 1

    def get_model(self):
        """
        Returns the model. Note: If you want to sample/infer from this,
        you will need to do `ema_helper.get_model().to('cuda')`.
        """
        return self.ema_model

