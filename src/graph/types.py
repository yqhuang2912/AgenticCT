import base64
from typing import Literal, Dict, List, Any, Optional
from pydantic import BaseModel
import torch
import io
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

from src.config import TEAM_MEMBERS

# Define routing options
OPTIONS = TEAM_MEMBERS + ["FINISH"]

class CTImageType(BaseModel):
    """Wrapper for torch.Tensor to ensure it can be serialized."""
    shape: tuple[int, ...]
    dtype: str
    data: str

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "CTImageType":
        """Convert a torch.Tensor to a CTImage."""
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return cls(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            data=data
        )
    
    def to_tensor(self) -> torch.Tensor:
        """Convert a CTImage back to a torch.Tensor."""
        data = base64.b64decode(self.data.encode('utf-8'))
        buffer = io.BytesIO(data)
        return torch.load(buffer)


class Router(TypedDict):
    """Router for supervisor node to decide next action."""
    next: Literal["ldct_processor", "svct_processor", "lact_processor", "FINISH"]  # Next action to take


class DegradationType(TypedDict):
    """Degradation type for CT image processing."""
    degradations: List[Dict[str, Any]]  # List of degradation parameters


class State(MessagesState):
    """State for the CT restoration agent system, extends MessagesState with CT-specific fields."""

    # Constants
    TEAM_MEMBERS: list[str]

    # Runtime Variables
    next: str
    full_plan: str
    current_step: Optional[int]  # Current processing step number (updated by processors)
    processing_count: Optional[int]  # Track number of processing iterations
    max_processing_iterations: Optional[int]  # Maximum allowed iterations (default: 5)
    remaining_steps: Optional[int]  # Remaining steps in the plan
    
    # Image data
    image_url: str  # Initial image URL
