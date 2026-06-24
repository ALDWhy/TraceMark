# watermarking/prompt_template.py

ANALYSIS_PROMPT = """你是一个专业的数字图像水印策略专家。请分析这张图片，并严格以纯 JSON 格式返回嵌入策略，不要包含任何 markdown 标记或额外解释。

返回格式要求：
{
    "content_type": "portrait" | "landscape" | "document" | "ai_generated" | "other",
    "sensitive_regions": ["face", "text", "signature", "none"], 
    "embed_strength": "high" | "medium" | "low",
    "extra_metadata": {"scene_desc": "简短场景描述(10字内)", "risk_level": "high|medium|low"}
}

分析规则：
1. 若包含清晰人脸或重要文字，embed_strength 必须为 "low"。
2. 若判定为 AI 生成图像或纹理复杂的风景，embed_strength 设为 "high"。
3. sensitive_regions 用于指导掩码的后处理保护。
"""