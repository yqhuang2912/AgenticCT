import logging
import torch
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

from .decorators import log_io
from src.tools.tool_call import get_tool
from src.utils.load_save_image import save_tensor_to_npy, load_npy_to_tensor, visualize_tensor

logger = logging.getLogger(__name__)



@tool
@log_io
def ldct_low_tool(
    image_url: Annotated[str, "Input low-dose CT image url"],
    current_step: Annotated[int, "Current processing step number"],
) -> Annotated[str, "Output reconstructed CT image url"]:
    """
    Low-level reconstruction tool for low-dose CT images.
    
    Args:
        image_url: Input low-dose CT image url
        current_step: Current processing step number
    Returns:
        Processed image url
    """
    logger.info(f"Running ldct_low_tool")
    model = get_tool("ldct", "low", "psnr")
    image = load_npy_to_tensor(image_url).to(next(model.parameters()).device)
    with torch.no_grad():
        processed_image_tensor = model(image)
    save_tensor_to_npy(processed_image_tensor, f"ldct_low_output_step_{current_step}.npy")
    visualize_tensor(processed_image_tensor, f"ldct_low_output_step_{current_step}.png")
    return f"ldct_low_output_step_{current_step}.npy"


@tool
@log_io
def ldct_medium_tool(
    image_url: Annotated[str, "Input low-dose CT image url"],
    current_step: Annotated[int, "Current processing step number"],
) -> Annotated[str, "Output reconstructed CT image url"]:
    """
    Medium-level reconstruction tool for low-dose CT images.
    
    Args:
        image_url: Input low-dose CT image url
        current_step: Current processing step number
    Returns:
        Processed image url
    """
    logger.info(f"Running ldct_medium_tool")
    model = get_tool("ldct", "medium", "psnr")
    image = load_npy_to_tensor(image_url).to(next(model.parameters()).device)
    with torch.no_grad():
        processed_image_tensor = model(image)
    save_tensor_to_npy(processed_image_tensor, f"ldct_medium_output_step_{current_step}.npy")
    visualize_tensor(processed_image_tensor, f"ldct_medium_output_step_{current_step}.png")
    return f"ldct_medium_output_step_{current_step}.npy"


@tool
@log_io
def ldct_high_tool(
    image_url: Annotated[str, "Input low-dose CT image url"],
    current_step: Annotated[int, "Current processing step number"],
) -> Annotated[str, "Output reconstructed CT image url"]:
    """
    High-level reconstruction tool for low-dose CT images.

    Args:
        image_url: Input low-dose CT image url
        current_step: Current processing step number
    Returns:
        Processed image url
    """
    logger.info(f"Running ldct_high_tool")
    model = get_tool("ldct", "high", "psnr")
    image = load_npy_to_tensor(image_url).to(next(model.parameters()).device)
    with torch.no_grad():
        processed_image_tensor = model(image)
    save_tensor_to_npy(processed_image_tensor, f"ldct_high_output_step_{current_step}.npy")
    visualize_tensor(processed_image_tensor, f"ldct_high_output_step_{current_step}.png")
    return f"ldct_high_output_step_{current_step}.npy"