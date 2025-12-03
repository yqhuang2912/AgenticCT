import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.prebuilt.chat_agent_executor import AgentState

from src.config.agents import LLMType

# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(prompt_name: str) -> str:
    """
    Load and return a prompt template using Jinja2.

    Args:
        prompt_name: Name of the prompt template file (without .md extension)

    Returns:
        The template string with proper variable substitution syntax
    """
    try:
        template = env.get_template(f"{prompt_name}.md")
        return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")


def apply_prompt_template(prompt_name: str, llm_type: LLMType, state: AgentState) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use
        llm_type: Type of LLM to use for the agent
        state: Current agent state containing variables to substitute

    Returns:
        List of messages with the system prompt as the first message
    """
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S"),
        **state,
    }

    try:
        template = env.get_template(f"{prompt_name}.md")
        system_prompt = template.render(**state_vars)
        
        # Debug logging for processor prompts
        if "processor" in prompt_name:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Template {prompt_name} - image_url: {state_vars.get('image_url', '[MISSING]')}")
            logger.debug(f"Template {prompt_name} - current_step: {state_vars.get('current_step', '[MISSING]')}")
        
        if llm_type == "vision":
            return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}] + state["messages"]
        else:
            return [{"role": "system", "content": system_prompt}] + state["messages"]
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")

