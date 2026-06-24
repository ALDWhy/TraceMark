# watermarking/llm_strategist.py
import os
import json
import base64
from typing import Dict, Any
from openai import OpenAI
from .prompt_template import ANALYSIS_PROMPT

DEFAULT_STRATEGY = {
    "content_type": "other",
    "sensitive_regions": [],
    "embed_strength": "medium",
    "extra_metadata": {"scene_desc": "未知", "risk_level": "medium"}
}

# ✅ 新增：合法策略的字段白名单与类型约束
VALID_STRENGTHS = {"low", "medium", "high"}
VALID_CONTENT_TYPES = {"portrait", "landscape", "document", "artwork", "other"}


class WatermarkStrategist:
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        # 优先使用百炼平台的环境变量，同时兼容旧的灵积平台变量
        self.api_key = api_key or os.getenv("BAILIAN_API_KEY") or os.getenv("QWEN_VL_API_KEY")
        if not self.api_key:
            raise ValueError("未设置 BAILIAN_API_KEY 或 QWEN_VL_API_KEY 环境变量")
        
        # 通过环境变量 BAILIAN_PLATFORM 明确指定平台
        # bailian: 阿里云百炼平台
        # dashscope: 阿里云灵积平台（兼容模式）
        platform = os.getenv("BAILIAN_PLATFORM", "").lower()
        
        if platform == "bailian" or os.getenv("BAILIAN_API_KEY"):
            # 百炼平台
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.baichuan-ai.com/v1")
            self.model_name = "Qwen-VL-Plus"
            print("[✅] 使用阿里云百炼平台")
        else:
            # 灵积平台（兼容模式）
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            self.model_name = "qwen-vl-max-latest"
            print("[✅] 使用阿里云灵积平台（兼容模式）")

    def _validate_strategy(self, raw: dict) -> dict:
        """✅ 校验 LLM 输出，非法字段回退默认值并记录警告"""
        validated = DEFAULT_STRATEGY.copy()
        warnings = []

        if isinstance(raw.get("embed_strength"), str) and raw["embed_strength"] in VALID_STRENGTHS:
            validated["embed_strength"] = raw["embed_strength"]
        else:
            warnings.append(f"embed_strength 非法: {raw.get('embed_strength')}")

        if isinstance(raw.get("content_type"), str) and raw["content_type"] in VALID_CONTENT_TYPES:
            validated["content_type"] = raw["content_type"]
        else:
            warnings.append(f"content_type 非法: {raw.get('content_type')}")

        if isinstance(raw.get("sensitive_regions"), list):
            validated["sensitive_regions"] = raw["sensitive_regions"]

        if isinstance(raw.get("extra_metadata"), dict):
            validated["extra_metadata"] = raw["extra_metadata"]

        if warnings:
            print(f"[⚠️ LLM策略校验] 以下字段已回退默认值: {'; '.join(warnings)}")

        return validated

    def analyze(self, image_path: str) -> Dict[str, Any]:
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": ANALYSIS_PROMPT}
                    ]
                }],
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()
            # 兼容 LLM 返回 markdown 代码块的情况
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            raw_strategy = json.loads(content)
            return self._validate_strategy(raw_strategy)

        except Exception as e:
            print(f"[⚠️ LLM决策降级] 解析失败: {e}，使用默认策略")
            return DEFAULT_STRATEGY.copy()