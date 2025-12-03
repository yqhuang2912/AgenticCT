import logging
import json
import torch
from pathlib import Path
import json_repair
import logging
from typing import Literal, Optional
from langchain_core.messages import HumanMessage, BaseMessage

import json_repair
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from PIL import Image

from src.agents import (
    ldct_processor_agent, 
    svct_processor_agent,
    lact_processor_agent,
    supervisor_agent,
)
from src.llms.llm import get_llm_by_type
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP
from src.prompts import apply_prompt_template
from src.utils.json_utils import repair_json_output
from .types import State

logger = logging.getLogger(__name__)

def coordinator_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    messages = apply_prompt_template("coordinator", AGENT_LLM_MAP["coordinator"], state)
    response = get_llm_by_type(AGENT_LLM_MAP["coordinator"]).invoke(messages)
    logger.debug(f"Current state messages: {state}")
    response_content = response.content
    # 尝试修复可能的JSON输出
    response_content = repair_json_output(response_content)
    logger.debug(f"Coordinator response: {response_content}")

    goto = "__end__"
    if "handoff_to_supervisor" in response_content:
        goto = "supervisor"

    response.content = response_content

    return Command(
        goto=goto,
    )

def supervisor_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Supervisor node that evaluates CT image degradations and decides next action."""
    logger.info("Supervisor evaluating degradation types and severity")

    # Check if maximum iterations reached
    processing_count = state["processing_count"]
    max_iterations = state["max_processing_iterations"]
    
    if processing_count >= max_iterations:
        logger.error(f"Maximum processing iterations ({max_iterations}) reached. Stopping workflow.")
        logger.error(f"Image may still have degradations but exceeded iteration limit.")
        return Command(
            goto="__end__",
            update={
                "messages": [HumanMessage(
                    content=f"ERROR: Maximum processing iterations ({max_iterations}) exceeded. Workflow terminated.",
                    name="supervisor"
                )]
            }
        )
    
    logger.info(f"Processing iteration: {processing_count + 1}/{max_iterations}")
    print("current state:", state)
    image_url = state.get("image_url", None)
    
    if not image_url:
        logger.error("No image found in state for degradation evaluation. Terminating workflow.")
        return Command(
            goto="__end__",
            update={
                "messages": [HumanMessage(
                    content="ERROR: No image found for degradation evaluation. Workflow terminated.",
                    name="supervisor"
                )]
            }
        )
    
    # Use a fresh state with only the current request to ensure the agent uses the current image_url
    # and isn't confused by previous history.
    invoke_state = state.copy()
    invoke_state["messages"] = [HumanMessage(content="Please evaluate all degradation types and their severity levels in this CT image.")]
    
    result = supervisor_agent.invoke(invoke_state)
    response = result["messages"][-1].content

    try:
        response_content = repair_json_output(response)
        logger.debug(f"Supervisor response: {response_content}")
        response_json = json_repair.loads(response_content)
    except json.JSONDecodeError:
        logger.error("Supervisor response is not a valid JSON")
        goto = "__end__"
        return Command(goto=goto)

    degradations = response_json["degradations"]
    if len(degradations) != 3:
        logger.error("Supervisor response does not contain all required degradation types")
        goto = "__end__"
        return Command(goto=goto)
    
    if all(severity.lower() == "none" for severity in degradations.values()):
        logger.info(f"No degradations detected - workflow completed successfully after {processing_count} iterations")
        goto = "__end__"
        return Command(goto=goto)
    
    else:
        # 如果是第一次处理，则交给planner节点生成完整计划
        if processing_count == 0:
            goto = "planner"
            state["current_step"] = 0
            return Command(
                goto=goto,
                update={
                    "messages": [HumanMessage(content=response_content, name="supervisor")],
                    "degradations": degradations,
                }
            )
        else:
            full_plan = json.loads(state.get("full_plan", "{}"))
            steps = full_plan.get("steps", [])
            idx = state.get("current_step", 0)
            if not steps or idx >= len(steps):
                goto = "supervisor"
            else:
                goto = steps[idx].get("processor_name", "supervisor")
            return Command(goto=goto)


def planner_node(state: State) -> Command[Literal["ldct_processor", "svct_processor", "lact_processor", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    messages = apply_prompt_template("planner", AGENT_LLM_MAP["planner"], state)
    llm = get_llm_by_type(AGENT_LLM_MAP["planner"])
    stream = llm.stream(messages)
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Planner response: {full_response}")

    if full_response.startswith("```json"):
        full_response = full_response.removeprefix("```json")

    if full_response.endswith("```"):
        full_response = full_response.removesuffix("```")

    try:
        plan = json_repair.loads(full_response)
        plan["current_step"] = 0
        steps = plan.get("steps", [])
        if not steps:
            full_response = json.dumps(plan)
            logger.debug(f"Repaired plan: {full_response}")
            return Command(
                update={
                    "messages": [HumanMessage(content=full_response, name="planner")],
                    "full_plan": full_response,
                },
                goto="supervisor",
            )
        goto = steps[0].get("processor_name", "supervisor")
        full_response = json.dumps(plan)
        logger.debug(f"Repaired plan: {full_response}")
        return Command(
            update={
                "messages": [HumanMessage(content=full_response, name="planner")],
                "full_plan": full_response,
            },
            goto=goto,
        )
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        goto = "__end__"
        return Command(goto=goto)   


def ldct_processor_node(state: State) -> Command[Literal["supervisor", "ldct_processor", "svct_processor", "lact_processor", "__end__"]]:
    """LDCT Processor node that processes low-dose CT images."""
    logger.info("LDCT Processor processing image")
    current_step = state["current_step"]
    next_step = current_step + 1
    logger.info(f"State current_step: current_step={current_step}")
    result = ldct_processor_agent.invoke(state)
    response_content = result["messages"][-1].content
    json_content = repair_json_output(response_content)
    content = json_repair.loads(json_content)
    
    if isinstance(content, str):
        try:
            content = json_repair.loads(content)
        except Exception:
            pass
            
    if not isinstance(content, dict):
        logger.error(f"LDCT Processor response is not a valid dict: {content}")
        processed_image_url = None
    else:
        processed_image_url = content.get("image_url", None)
        
    logger.debug(f"ldct_processor_node processed_image_url: {processed_image_url}")

    if not processed_image_url:
        logger.error("No image found in state for LDCT processing. Terminating workflow.")
        return Command(
            goto="__end__",
            update={
                "messages": [HumanMessage(
                    content="ERROR: No image found for LDCT processing. Workflow terminated.",
                    name="ldct_processor"
                )]
            }
        )

    full_plan = json.loads(state.get("full_plan", "{}"))
    if next_step >= len(full_plan.get("steps", [])):
        goto = "supervisor"
    else:
        goto = full_plan["steps"][next_step]["processor_name"]
    
    return Command(
        goto=goto,
        update={
            "image_url": processed_image_url,
            "current_step": next_step,
            "processing_count": state["processing_count"] + 1,
        }
    )


def svct_processor_node(state: State) -> Command[Literal["supervisor", "ldct_processor", "svct_processor", "lact_processor"]]:
    """SVCT Processor node that processes sparse-view CT images."""
    logger.info("SVCT Processor processing image")
    current_step = state["current_step"]
    next_step = current_step + 1
    logger.info(f"State current_step: current_step={current_step}")
    result = svct_processor_agent.invoke(state)
    response_content = result["messages"][-1].content
    json_content = repair_json_output(response_content)
    content = json_repair.loads(json_content)
    
    if isinstance(content, str):
        try:
            content = json_repair.loads(content)
        except Exception:
            pass
            
    if not isinstance(content, dict):
        logger.error(f"SVCT Processor response is not a valid dict: {content}")
        processed_image_url = None
    else:
        processed_image_url = content.get("image_url", None)
        
    logger.debug(f"svct_processor_node processed_image_url: {processed_image_url}")

    if not processed_image_url:
        logger.error("No image found in state for SVCT processing. Terminating workflow.")
        return Command(
            goto="__end__",
            update={
                "messages": [HumanMessage(
                    content="ERROR: No image found for SVCT processing. Workflow terminated.",
                    name="svct_processor"
                )]
            }
        )
    
    full_plan = json.loads(state.get("full_plan", "{}"))
    if next_step >= len(full_plan.get("steps", [])):
        goto = "supervisor"
    else:
        goto = full_plan["steps"][next_step]["processor_name"]
    
    return Command(
        goto=goto,
        update={
            "image_url": processed_image_url,
            "current_step": next_step,
            "processing_count": state["processing_count"] + 1,
        }
    )

def lact_processor_node(state: State) -> Command[Literal["supervisor", "ldct_processor", "svct_processor", "lact_processor"]]:
    """LACT Processor node that processes limited-angle CT images."""
    logger.info("LACT Processor processing image")
    current_step = state["current_step"]
    next_step = current_step + 1
    logger.info(f"State current_step: current_step={current_step}") 
    result = lact_processor_agent.invoke(state) # json type
    
    response_content = result["messages"][-1].content
    json_content = repair_json_output(response_content)
    content = json_repair.loads(json_content)
    
    if isinstance(content, str):
        try:
            content = json_repair.loads(content)
        except Exception:
            pass
            
    if not isinstance(content, dict):
        logger.error(f"LACT Processor response is not a valid dict: {content}")
        processed_image_url = None
    else:
        processed_image_url = content.get("image_url", None)
        
    logger.debug(f"lact_processor_node processed_image_url: {processed_image_url}")

    if not processed_image_url:
        logger.error("No image found in state for LACT processing. Terminating workflow.")
        return Command(
            goto="__end__",
            update={
                "messages": [HumanMessage(
                    content="ERROR: No image found for LACT processing. Workflow terminated.",
                    name="lact_processor"
                )]
            }
        )

    full_plan = json.loads(state.get("full_plan", "{}"))
    if next_step >= len(full_plan.get("steps", [])):
        goto = "supervisor"
    else:
        goto = full_plan["steps"][next_step]["processor_name"]
    
    return Command(
        goto=goto,
        update={
            "image_url": processed_image_url,
            "current_step": next_step,
            "processing_count": state["processing_count"] + 1,
        }
    )

    full_plan = json.loads(state.get("full_plan", "{}"))
    if next_step >= len(full_plan.get("steps", [])):
        goto = "supervisor"
    else:
        goto = full_plan["steps"][next_step]["processor_name"]
    
    return Command(
        goto=goto,
        update={
            "image_url": processed_image_url,
            "current_step": next_step,
            "processing_count": state["processing_count"] + 1,
        }
    )
