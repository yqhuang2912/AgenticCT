import torch
import numpy as np
from PIL import Image

def save_tensor(tensor: torch.Tensor, file_path: str):
    """
    Save a torch tensor as an image file.
    
    Args:
        tensor (torch.Tensor): The tensor to save. Expected shape is [1, 1, H, W].
        file_path (str): The path where the image will be saved.
        amin (int): Minimum value of the original pixel range.
        amax (int): Maximum value of the original pixel range.
        vmin (int): Minimum value of the display window.
        vmax (int): Maximum value of the display window.
    """
    # Ensure tensor is on CPU and convert to numpy
    print("tensor shape:", tensor.shape)
    tensor = tensor.squeeze().cpu().numpy() # range(0.0, 1.0)
    # Scale to [amin, amax]
    tensor = tensor * 255.0  # Assuming input tensor is normalized to [0, 1]
    tensor = tensor.clip(0, 255)

    # Convert to uint8
    tensor = tensor.astype('uint8')

    # Convert to PIL Image and save
    img = Image.fromarray(tensor)
    img.save(file_path)


def load_image(image_url: str, target_shape: tuple = None, typ: str = "normal"):
    """Load and preprocess an image.

    Args:
        image_url (str): Path to the image file.
        target_shape (tuple, optional): Desired shape (height, width) to resize the image. Defaults to None.
        typ (str): Type of preprocessing ('normal' | 'deeplesion' | 'mayo'). Defaults to 'normal'.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if typ == "normal":
        img = Image.open(image_url).convert("L")  # shape: (H, W)
        if target_shape is not None:
            img = img.resize(target_shape, Image.BILINEAR)
        img = np.asarray(img).astype(np.float32)  # Normalize to [0, 1]
    elif typ == "deeplesion":
        img = Image.open(image_url)
        if target_shape is not None:
            img = img.resize(target_shape, Image.BILINEAR)
        img = np.asarray(img).astype(np.float32) - 32768.0
        min_hu, max_hu = -1024.0, 1024.0
        img = np.clip(img, min_hu, max_hu)
        img = (img - min_hu) / (max_hu - min_hu) # Normalize to [0, 1]
        img = img.astype(np.float32) * 255.0
    elif typ == 'mayo':
        pass
    else:
        raise ValueError("Unsupported preprocessing type.")
    return torch.tensor(img).unsqueeze(0).unsqueeze(0) # shape: (1, 1, H, W)


