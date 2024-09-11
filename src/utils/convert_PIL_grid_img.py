import torchvision
import wandb
import torch
import numpy as np
from PIL import Image

def convert_PIL_grid_img(batch_imgs, target_channel=None, nrow=5):
    grid_tensor = torchvision.utils.make_grid(batch_imgs.permute(0,3,1,2), nrow=nrow)
    grid_tensor = grid_tensor.permute(1,2,0)
    # grid_tensor = torch.ones(grid_tensor.shape[:3]) - grid_tensor
    if target_channel == None:
        return wandb.Image((grid_tensor * 255).cpu().numpy().astype(np.uint8))
    else:
        return wandb.Image((grid_tensor[...,target_channel] * 255).cpu().numpy().astype(np.uint8))
