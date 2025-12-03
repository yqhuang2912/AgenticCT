from typing import Literal

LLMType = Literal["basic", "vision"]

# CT图像修复系统智能体配置
# 配置了多种专业智能体来处理不同类型的CT图像退化问题

AGENT_LLM_MAP: dict[str, LLMType] = {
    # 核心协调智能体
    "coordinator": "basic",                    # 总协调者，负责整个流程的控制
    "supervisor": "basic",                     # 检测降质类型，选择合适的处理智能体
    "planner": "basic",                        # 策略规划者，制定多退化处理策略

    # 检测和评估智能体
    "ldct_processor": "basic",                 # 低剂量CT处理器，处理低剂量噪声
    "svct_processor": "basic",                 # 稀疏角CT处理器
    "lact_processor": "basic",                 # 有限角CT处理器

}