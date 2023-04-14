import torchvision
import wandb
import numpy as np

def convert_PIL_grid_img(batch_imgs, target_channel, nrow):
    grid_tensor = torchvision.utils.make_grid(batch_imgs.permute(0,3,1,2), nrow=nrow)
    grid_tensor = grid_tensor.permute(1,2,0)
    if not target_channel:
        return wandb.Image((grid_tensor * 255).cpu().numpy().astype(np.uint8))
    else:
        return wandb.Image((grid_tensor[...,target_channel] * 255).cpu().numpy().astype(np.uint8))
