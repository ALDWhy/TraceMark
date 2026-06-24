# tamper_detector.py - AI 图像篡改检测模块
import os
import json
import base64
from typing import Dict, Any, Tuple
from openai import OpenAI

DEFAULT_RESULT = {
    "is_tampered": False,
    "confidence": 0.0,
    "tamper_type": "none",
    "tamper_regions": [],
    "description": "图像完整，未检测到篡改"
}

TAMPER_TYPES = {
    "none": "未检测到篡改",
    "splicing": "图像拼接/合成",
    "retouching": "局部修饰/PS",
    "filter": "滤镜/调色修改",
    "compression": "过度压缩",
    "unknown": "疑似篡改，类型不确定"
}


class TamperDetector:
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

    def _build_prompt(self) -> str:
        """构建篡改检测的提示词"""
        return """你是一个专业的数字图像真伪鉴定专家。请仔细分析这张图片，判断是否被篡改过。

分析维度：
1. 图像拼接痕迹：不同区域的光照、色彩、噪点是否一致
2. 边缘异常：是否有生硬的边界或过渡不自然的区域
3. 像素异常：局部区域的像素分布是否异常
4. 压缩痕迹：是否有过度压缩或多次压缩的痕迹
5. AI 生成特征：是否存在 AI 生成图像的典型特征

请严格按照以下 JSON 格式返回分析结果：
{
    "is_tampered": true/false,
    "confidence": 0.0-1.0,
    "tamper_type": "none|splicing|retouching|filter|compression|unknown",
    "tamper_regions": ["区域描述，如：左上角", "区域描述"]或[],
    "description": "详细分析说明，100字以内"
}

注意：只返回纯 JSON，不要包含其他任何文字。
"""

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        使用 AI 检测图像是否被篡改
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            检测结果字典
        """
        try:
            # 读取并编码图像
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            # 构建请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": self._build_prompt()}
                    ]
                }],
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()
            
            # 处理可能的 markdown 代码块
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(content)
            return self._validate_result(result)

        except Exception as e:
            print(f"[⚠️ AI篡改检测失败] {e}，使用默认结果")
            return DEFAULT_RESULT.copy()

    def _validate_result(self, raw: dict) -> Dict[str, Any]:
        """校验 AI 返回结果的合法性"""
        validated = DEFAULT_RESULT.copy()

        # 校验 is_tampered
        if isinstance(raw.get("is_tampered"), bool):
            validated["is_tampered"] = raw["is_tampered"]

        # 校验 confidence
        confidence = raw.get("confidence")
        if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
            validated["confidence"] = confidence

        # 校验 tamper_type
        tamper_type = raw.get("tamper_type")
        if tamper_type in TAMPER_TYPES:
            validated["tamper_type"] = tamper_type
            validated["description"] = TAMPER_TYPES[tamper_type]

        # 校验 tamper_regions
        if isinstance(raw.get("tamper_regions"), list):
            validated["tamper_regions"] = raw["tamper_regions"]

        # 使用自定义描述（如果有）
        if isinstance(raw.get("description"), str):
            validated["description"] = raw["description"]

        return validated


# 简化的函数接口（方便直接调用）
def detect_tampering(image_path: str) -> Dict[str, Any]:
    """检测图像篡改的便捷函数"""
    try:
        detector = TamperDetector()
        return detector.detect(image_path)
    except Exception as e:
        print(f"[⚠️ 检测初始化失败] {e}")
        return DEFAULT_RESULT.copy()


# 使用示例
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python tamper_detector.py <image_path>")
        sys.exit(1)
    
    result = detect_tampering(sys.argv[1])
    print("=== AI 篡改检测结果 ===")
    print(f"是否篡改: {'是' if result['is_tampered'] else '否'}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"篡改类型: {result['tamper_type']}")
    print(f"可疑区域: {result['tamper_regions']}")
    print(f"描述: {result['description']}")