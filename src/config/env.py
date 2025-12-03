import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Reasoning model configuration
BASIC_MODEL = os.getenv("BASIC_MODEL", "deepseek-chat")
BASIC_BASE_URL = os.getenv("BASIC_BASE_URL", "https://api.deepseek.com")
BASIC_API_KEY = os.getenv("BASIC_API_KEY")

VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max-latest")
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VISION_API_KEY = os.getenv("VISION_API_KEY")