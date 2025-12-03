from .env import (
    # Basic LLM
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    # Vision LLM
    VISION_MODEL,
    VISION_BASE_URL,
    VISION_API_KEY,
)

# Team configuration - CT Multi-Degradation Detection and Restoration Agents
TEAM_MEMBER_CONFIGRATIONS = {
    "ldct_processor": {
        "name": "ldct_processor",
        "desc": (
            "低剂量CT处理专家，专门处理低剂量CT图像的噪声问题，支持RED-CNN方法，可以根据不同的噪声水平（low,medium,high）进行处理"
        ),
        "is_optional": False,
    },
    "svct_processor": {
        "name": "svct_processor",
        "desc": (
            "去噪专家，专门处理稀疏角CT图像的噪声问题，支持FBPConvNet方法, 可以根据不同的稀疏程度（low,medium,high）进行处理"
        ),
        "is_optional": False,
    },
    "lact_processor": {
        "name": "lact_processor",
        "desc": (
            "有限角CT处理专家，专门处理有限角CT图像的噪声问题，支持FBPConvNet方法, 可以根据不同的稀疏程度（low,medium,high）进行处理"
        ),
        "is_optional": False,
    },
}

TEAM_MEMBERS = list(TEAM_MEMBER_CONFIGRATIONS.keys())

__all__ = [
    # Basic LLM
    "BASIC_MODEL",
    "BASIC_BASE_URL",
    "BASIC_API_KEY",
    # Vision LLM
    "VISION_MODEL",
    "VISION_BASE_URL",
    "VISION_API_KEY",
]
