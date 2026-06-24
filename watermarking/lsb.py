# lsb.py - 支持 LLM 策略驱动的自适应 LSB 嵌入
from PIL import Image

# ✅ LLM 策略到 LSB 嵌入模式的映射
# low: 仅 R 通道 LSB（最小改动，最高画质）
# medium: RGB 三通道 LSB（默认，平衡）
# high: RGB 三通道 LSB + G 通道次低位（冗余嵌入，最强鲁棒性）
STRENGTH_MODE_MAP = {
    "low": {"channels": ["R"], "bit_planes": [0]},
    "medium": {"channels": ["R", "G", "B"], "bit_planes": [0]},
    "high": {"channels": ["R", "G", "B"], "bit_planes": [0, 1]},
}


def embed_uid_in_image(
        input_path: str,
        output_path: str,
        uid: str,
        *,
        strength: str = "medium",
        content_type: str = "other",
        sensitive_regions: list = None,
        **kwargs
):
    """
    将 UID 嵌入 PNG 图像 LSB 位。

    Args:
        strength: LLM 决策的嵌入强度 ("low"/"medium"/"high")
        content_type: LLM 识别的内容类型
        sensitive_regions: 敏感区域列表（预留）
    """
    mode = STRENGTH_MODE_MAP.get(strength, STRENGTH_MODE_MAP["medium"])
    channels = mode["channels"]
    bit_planes = mode["bit_planes"]
    print(f"[LSB] 嵌入参数: strength={strength}, channels={channels}, bit_planes={bit_planes}, content_type={content_type}")

    img = Image.open(input_path).convert("RGB")
    pixels = list(img.getdata())

    data = f"UID:{uid}\x00"
    byte_data = data.encode('utf-8')
    binary = ''.join(format(byte, '08b') for byte in byte_data)

    bits_per_pixel = len(channels) * len(bit_planes)
    capacity = len(pixels) * bits_per_pixel

    if len(binary) > capacity:
        raise ValueError(
            f"图像容量不足: 需要 {len(binary)} bits，"
            f"当前模式({strength})仅支持 {capacity} bits。"
            f"建议提高嵌入强度或更换更大图像。"
        )

    channel_map = {"R": 0, "G": 1, "B": 2}
    new_pixels = []
    bit_idx = 0

    for pixel in pixels:
        pixel_list = list(pixel)
        for ch_name in channels:
            ch_idx = channel_map[ch_name]
            for bp in bit_planes:
                if bit_idx < len(binary):
                    mask = ~(1 << bp) & 0xFF
                    pixel_list[ch_idx] = (pixel_list[ch_idx] & mask) | (int(binary[bit_idx]) << bp)
                    bit_idx += 1
        new_pixels.append(tuple(pixel_list))

    img.putdata(new_pixels)
    img.save(output_path, "PNG")


def extract_uid_from_image(img_path: str) -> str:
    """
    提取 UID。
    """
    img = Image.open(img_path)
    pixels = list(img.getdata())

    bits = ""
    max_bits = min(len(pixels) * 3, 10000)

    for r, g, b in pixels:
        if len(bits) >= max_bits:
            break
        bits += str(r & 1)
        bits += str(g & 1)
        bits += str(b & 1)

    full_bytes = len(bits) // 8
    all_bytes = bytearray()
    for i in range(full_bytes):
        byte_val = int(bits[i * 8:(i + 1) * 8], 2)
        all_bytes.append(byte_val)

    null_index = all_bytes.find(b'\x00')
    if null_index == -1:
        raise ValueError("未找到终止符 \\x00，水印可能不完整")

    byte_vals = all_bytes[:null_index]
    try:
        result = byte_vals.decode('utf-8')
        if result.startswith("UID:"):
            return result[4:]
        else:
            raise ValueError("水印格式错误：缺少 'UID:' 前缀")
    except UnicodeDecodeError:
        raise ValueError("无效的 UTF-8 数据，非本系统水印")