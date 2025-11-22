# Scale to [-1, 1] (diffusion models usually expect this)
def scale_to_minus_one_to_one(x):
    return x * 2. - 1.
LR=64
HR=256
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LRHRDataset(Dataset):
    """
    Wraps an existing dataset that returns [3,H,H] tensors in [-1,1],
    and produces:
      - hr: [3, HR, HR]
      - lr: [3, LR, LR]
      #- cond_lr_up [3, HR, HR] (LR downsampled to LR_SIZE then upsampled)
    """
    def __init__(self, hr_dataset, lr_size=LR, hr_size=HR):
        super().__init__()
        self.hr_dataset = hr_dataset
        self.lr_size = lr_size
        self.hr_size = hr_size

    def __len__(self):
        return len(self.hr_dataset)

    def __getitem__(self, idx):
        hr_img, _ = self.hr_dataset[idx]  # [3,HR,HR]
        # Ensure it's [3,HR,HR]
        if hr_img.shape[1] != self.hr_size:
            print("Would be faster to use a pre-scaled dataset")
            hr_img = F.interpolate(hr_img.unsqueeze(0), size=(self.hr_size, self.hr_size),
                                mode='bicubic', align_corners=False, antialias=True).squeeze(0)
        # Create LR then upscale
        lr_img = F.interpolate(hr_img.unsqueeze(0), size=(self.lr_size, self.lr_size),
                               mode='bicubic', align_corners=False, antialias=True)
        #cond_lr_up = F.interpolate(lr_img, size=(self.hr_size, self.hr_size),
        #                           mode='bilinear', align_corners=False).squeeze(0)
        #return hr_img, cond_lr_up
        return hr_img, lr_img.squeeze(0)
    
def get_datasets(experiment_name,LR=LR,HR=HR):
    lr_transform = transforms.Compose([
            transforms.Resize((LR, LR)),
            transforms.ToTensor(),
            transforms.Lambda(scale_to_minus_one_to_one),
    ])
    hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(scale_to_minus_one_to_one),
    ])
    hr_dataset = datasets.ImageFolder(root=f"data/256x256/{experiment_name}",transform=hr_transform)
    lr_dataset = datasets.ImageFolder(root=f"data/256x256/{experiment_name}",transform=lr_transform)
    paired_dataset = LRHRDataset(hr_dataset)
    return lr_dataset,paired_dataset


