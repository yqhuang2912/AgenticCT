import numpy as np
import torch
import wandb
from torchvision.utils import make_grid
from utils.utils import denormalize_image, get_adaptive_window, normalize_image

def calc_vmin_vmax(window_center, window_width, select=0):
    vmin = window_center[select] - 0.5 * window_width[select]
    vmax = window_center[select] + 0.5 * window_width[select]
    return vmin, vmax

def denormalize(x, min_hu=-1024.0, max_hu=3072.0):
    x = x * (max_hu - min_hu) + min_hu
    return np.clip(x, min_hu, max_hu)

def clip_to_window(x, vmin=-160, vmax=240):
    """
    Normalize the input tensor x based on the specified window.
    """
    x = denormalize(x)
    x[x < vmin] = vmin
    x[x > vmax] = vmax
    return x

def show_win_norm(x, vmin=-160, vmax=240):
    x = (x - vmin) / (vmax - vmin) * 255
    return x

def log_images_to_wandb(x, y, out, mode):
    """Log images to wandb for visualization"""
    
    x = x[0].detach().cpu().numpy()
    out = out[0].detach().cpu().numpy()
    y = y[0].detach().cpu().numpy()

    x_vmin, x_vmax = get_adaptive_window(x_vis)
    x_vis = normalize_image(x_vis, x_vmin, x_vmax) * 255
    out_vmin, out_vmax = get_adaptive_window(out_vis)
    out_vis = normalize_image(out_vis, out_vmin, out_vmax) * 255
    y_vmin, y_vmax = get_adaptive_window(y_vis)
    y_vis = normalize_image(y_vis, y_vmin, y_vmax) * 255

    wandb.log({f"{mode}_images": wandb.Image(x_vis, caption="Input Degraded Image")}, commit=False)
    wandb.log({f"{mode}_images": wandb.Image(out_vis, caption="Output Image")}, commit=False)
    wandb.log({f"{mode}_images": wandb.Image(y_vis, caption="Ground Truth Image")}, commit=False)

def log_metrics_to_wandb(metrics, mode='train'):
    """Log metrics to wandb"""
    for key, value in metrics.items():
        wandb.log({f"{mode}_{key}": value})