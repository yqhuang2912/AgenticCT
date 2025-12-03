from langgraph.graph import StateGraph, START
from .types import State
from .nodes import (
    coordinator_node, 
    planner_node, supervisor_node, 
    ldct_processor_node,
    svct_processor_node, 
    lact_processor_node, 
)


def build_graph():
    """Build and return the CT restoration agent workflow graph."""
    builder = StateGraph(State)
    
    # 起始流程
    builder.add_edge(START, "coordinator")
    
    # 添加所有节点
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("planner", planner_node)
    
    # CT专用节点
    builder.add_node("ldct_processor", ldct_processor_node)
    builder.add_node("svct_processor", svct_processor_node)
    builder.add_node("lact_processor", lact_processor_node)
    return builder.compile()