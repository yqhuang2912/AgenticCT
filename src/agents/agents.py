from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.graph.types import State  # Import our State type
from src.tools import (
    ldct_low_tool,
    ldct_medium_tool,
    ldct_high_tool,
    svct_low_tool,
    svct_medium_tool,
    svct_high_tool,
    lact_low_tool,
    lact_medium_tool,
    lact_high_tool,
    ctqe_tool,
)

from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP

supervisor_agent = create_react_agent(
    name="supervisor",
    model=get_llm_by_type(AGENT_LLM_MAP["supervisor"]),
    tools=[ctqe_tool],
    prompt=lambda state: apply_prompt_template("supervisor", AGENT_LLM_MAP["supervisor"], state),
    state_schema=State,
)

ldct_processor_agent = create_react_agent(
    name="ldct_processor",
    model=get_llm_by_type(AGENT_LLM_MAP["ldct_processor"]),
    tools=[ldct_low_tool, ldct_medium_tool, ldct_high_tool],
    prompt=lambda state: apply_prompt_template("ldct_processor", AGENT_LLM_MAP["ldct_processor"], state),
    state_schema=State,
)

svct_processor_agent = create_react_agent(
    name="svct_processor",
    model=get_llm_by_type(AGENT_LLM_MAP["svct_processor"]),
    tools=[svct_low_tool, svct_medium_tool, svct_high_tool],
    prompt=lambda state: apply_prompt_template("svct_processor", AGENT_LLM_MAP["svct_processor"], state),
    state_schema=State,
)

lact_processor_agent = create_react_agent(
    name="lact_processor",
    model=get_llm_by_type(AGENT_LLM_MAP["lact_processor"]),
    tools=[lact_low_tool, lact_medium_tool, lact_high_tool],
    prompt=lambda state: apply_prompt_template("lact_processor", AGENT_LLM_MAP["lact_processor"], state),
    state_schema=State,
)