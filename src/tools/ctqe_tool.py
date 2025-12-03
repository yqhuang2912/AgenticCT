import logging
from unittest import result
import torch
from pathlib import Path
from typing import Annotated, Tuple, Dict
import json
import torch.nn.functional as F

from langchain_core.tools import tool

from .decorators import log_io
from src.tools.tool_call import get_tool
from src.utils.load_save_image import load_npy_to_tensor

logger = logging.getLogger(__name__)


LABELS = ["none", "low", "medium", "high"]


def _decode_head(logits: torch.Tensor) -> str:
    """logits [1,4] -> (label, confidence, probs_dict)."""
    probs = F.softmax(logits, dim=-1)[0]  # [4]
    idx = int(torch.argmax(probs).item())
    label = LABELS[idx]
    return label


@tool
@log_io
def ctqe_tool(
    image_url: Annotated[str, "Input CT image (local .npy path)"],
) -> Annotated[str, "json string of quality assessment results"]:
    """Predict degradation severities for (ldct, lact, svct) using CTQEModel."""
    logger.info("Running ctqe_tool...")

    model = get_tool("ctqe", severity="", metric="acc")

    x = load_npy_to_tensor(image_url).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(x)

    ldct_label = _decode_head(out["ldct"])
    lact_label = _decode_head(out["lact"])
    svct_label = _decode_head(out["svct"])

    result = {
        "degradations": {"ldct": ldct_label, "lact": lact_label, "svct": svct_label},
    }
    return json.dumps(result, ensure_ascii=False)
