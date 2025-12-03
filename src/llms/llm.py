from pyexpat.errors import messages
import unsloth
import torch
from langchain_deepseek import ChatDeepSeek
from typing import Any, Optional
from unsloth import FastVisionModel
from PIL import Image
from src.config.agents import LLMType
from src.utils.load_save_image import load_npy_to_tensor, convert_to_image_range, get_adaptive_window, denormalize_image

class UnslothCustomLLM:
    def __init__(self, 
                 model_dir: str,
                 temperature: float = 0.0,
                 max_new_tokens: int = 512,
                 use_cache: bool = True,
                 min_p: float = 0.1
    ):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.min_p = min_p
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
            local_files_only=True
        )
        FastVisionModel.for_inference(self.model)
    
    def _build_inputs(self, image_url, messages) -> str:
        image = load_npy_to_tensor(image_url)
        image = denormalize_image(image.squeeze().numpy(), min_hu=-1024, max_hu=3072)
        vmin, vmax = get_adaptive_window(image)
        image = convert_to_image_range(image, vmin, vmax)
        import matplotlib.pyplot as plt
        plt.imshow(image, cmap='gray')
        plt.savefig("debug_image.png", dpi=300)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            text=input_text,
            images=image,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.model.device)

        return model_inputs

    def invoke(self, image_url, messages) -> str:
        inputs = self._build_inputs(image_url, messages)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            use_cache=self.use_cache,
            min_p=self.min_p,
        )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()



def create_deepseek_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatDeepSeek:
    """
    Create a ChatDeepSeek instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["base_url"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatDeepSeek(**llm_kwargs)


def create_vision_llm(
    model_dir: str,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    use_cache: bool = True,
    min_p: float = 0.1,
) -> UnslothCustomLLM:
    """
    Create a UnslothCustomLLM instance with the specified configuration
    """
    return UnslothCustomLLM(
        model_dir=model_dir,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        min_p=min_p,
    )


_llm_cache: dict[LLMType: UnslothCustomLLM | ChatDeepSeek] = {}

def get_llm_by_type(llm_type: LLMType) -> UnslothCustomLLM | ChatDeepSeek:
    """
    Factory function to get LLM instances by type.
    Caches instances to avoid redundant creations.
    """
    if llm_type not in _llm_cache:
        if llm_type == "vision":
            llm_instance = create_vision_llm(
                model_dir="/data/hyq/pretrained/Qwen/Qwen3-VL-8B-Instruct",
                # model_dir="/data/hyq/projects/unsloth_qwen3_vl_8b_sft_ct/exp_20251112_025719",
                temperature=0.7,
                max_new_tokens=256,
                use_cache=True,
                min_p=0.05,
            )
            _llm_cache[llm_type] = llm_instance
        elif llm_type == "basic":
            llm_instance = create_deepseek_llm(
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key="sk-eb8a69d0556f4581a28290dec24d7798",
            )
            _llm_cache[llm_type] = llm_instance
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    return _llm_cache[llm_type]


if __name__ == "__main__":
    # Test the custom LLM with an example

    SYSTEM_PROMPT = """You are an expert in CT image quality assessment.
Your task is to analyze CT images and identify the type and severity of image degradation present.

Possible degradation type:
- "low-dose" (caused by reduced radiation dose)

For the identified degradation type, estimate its severity on a continuous scale from 0 to 1,
where 0 = no degradation and 1 = extremely severe degradation.

Return your answer strictly in JSON format, like this:
```json
{
  "degradations": [{"type": "low-dose", "severity": 0.40}]
}
```
If no degradation is detected, return:
```json 
{
  "degradations": "none"
}
```
"""
    model_dir = "/data/hyq/pretrained/Qwen/Qwen3-VL-8B-Instruct"
    user_query = "Please evaluate all degradation type and their severity level in this CT image."
    image_url = "/data/hyq/codes/ct_restore_agent/la.png"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {"type": "image"}
            ]
        }
    ]

    response = get_llm_by_type("vision").invoke(
            image_url=image_url,
            messages=messages
        )
    print("Custom LLM Response:", response)

    # Test the DeepSeek LLM with an example
    deepseek_llm = get_llm_by_type("basic")
    stream = deepseek_llm.stream(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
    print("DeepSeek LLM Response:", full_response)