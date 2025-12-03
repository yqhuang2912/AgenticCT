import torch
import logging
from src.pretrain.processors.models.redcnn import RED_CNN
from src.pretrain.processors.models.fbpconvnet import FBPConvNet
from src.pretrain.supervisor.models.ctqe_model import CTQEModel

PRETRAINED_MODEL_BASE_DIR = "/data/hyq/codes/AgenticCT/src/pretrain/outputs"
logger = logging.getLogger(__name__)

def ctqe_tool(model_path: str, device: str='cuda') -> CTQEModel:
    """
    CT 图像质量评估工具。
    Args:
        model_path: 预训练模型路径
        device: 运行设备 ("cpu" 或 "cuda")
    Returns:
        加载了预训练权重的模型实例
    """
    model_instance = CTQEModel(pretrained=False, num_classes=4)
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.to(device)
    model_instance.eval()
    return model_instance


def lact_tool(model_path: str, device: str='cuda') -> FBPConvNet:
    """
    Limited-angle CT 图像重建工具。
    Args:
        model_path: 预训练模型路径
        device: 运行设备 ("cpu" 或 "cuda")
    Returns:
        加载了预训练权重的模型实例
    """
    model_instance = FBPConvNet()
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.to(device)
    model_instance.eval()
    return model_instance


def ldct_tool(model_path: str, device: str='cuda') -> RED_CNN:
    """
    Low-dose CT 图像重建工具。
    Args:
        model_path: 预训练模型路径
        device: 运行设备 ("cpu" 或 "cuda")
    Returns:
        加载了预训练权重的模型实例
    """
    model_instance = RED_CNN()
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.to(device)
    model_instance.eval()
    return model_instance


def svct_tool(model_path: str, device: str='cuda') -> FBPConvNet:
    """
    Sparse-view CT 图像重建工具。
    Args:
        model_path: 预训练模型路径
        device: 运行设备 ("cpu" 或 "cuda")
    Returns:
        加载了预训练权重的模型实例
    """
    model_instance = FBPConvNet()
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.to(device)
    model_instance.eval()
    return model_instance


_tool_cache: dict[str: RED_CNN | FBPConvNet] = {}

def get_tool(tool_name: str, severity: str, metric: str="psnr") -> RED_CNN | FBPConvNet:
    key = f"{tool_name}_{severity}_{metric}"
    if key in _tool_cache:
        logger.debug(f"Using cached model for {key}")
        return _tool_cache[key]

    if tool_name == "ldct":
        model_path = f"{PRETRAINED_MODEL_BASE_DIR}/ldct/{severity}/best_val_{metric}_model.pth"
        model_instance = ldct_tool(model_path)
    elif tool_name == "lact":
        model_path = f"{PRETRAINED_MODEL_BASE_DIR}/lact/{severity}/best_val_{metric}_model.pth"
        model_instance = lact_tool(model_path)
    elif tool_name == "svct":
        model_path = f"{PRETRAINED_MODEL_BASE_DIR}/svct/{severity}/best_val_{metric}_model.pth"
        model_instance = svct_tool(model_path)
    elif tool_name == "ctqe":
        model_path = f"{PRETRAINED_MODEL_BASE_DIR}/ctqe/default/best_val_{metric}_model.pth"
        model_instance = ctqe_tool(model_path)
    else:
        raise ValueError(f"Unknown tool name: {tool_name}")

    _tool_cache[key] = model_instance
    logger.debug(f"Loaded model for {key} from {model_path}")
    return model_instance
    