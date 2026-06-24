# dct.py - 统一接口版（支持 LLM 策略驱动）
import cv2
import numpy as np
from scipy.fft import dct, idct

# 默认配置（当 LLM 未指定或降级时使用）
DEFAULT_EMBED_POS = (4, 4)
DEFAULT_THRESHOLD = 8

# ✅ LLM 策略到工程参数的映射表
STRENGTH_MAP = {
    "low": 4,      # 低强度：视觉质量优先，鲁棒性较弱
    "medium": 8,   # 中强度：平衡（默认）
    "high": 16     # 高强度：鲁棒性优先，可能有轻微失真
}


def _encode_uid_to_bits(uid: str) -> str:
    """将 UID 转为带终止符的二进制字符串（UTF-8）"""
    data = f"UID:{uid}\x00"
    byte_data = data.encode('utf-8')
    return ''.join(format(byte, '08b') for byte in byte_data)


def _decode_bits_to_uid(bits: str) -> str:
    """从二进制字符串还原 UID（UTF-8 + 终止符处理）"""
    byte_vals = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 > len(bits):
            break
        byte_val = int(bits[i:i+8], 2)
        if byte_val == 0:
            break
        byte_vals.append(byte_val)

    result = byte_vals.decode('utf-8')
    if result.startswith("UID:"):
        return result[4:]
    else:
        raise ValueError("水印格式错误：缺少 'UID:' 前缀")


def embed_uid_in_image(
        input_path: str,
        output_path: str,
        uid: str,
        *,                          # ✅ 强制关键字参数，避免与旧调用混淆
        strength: str = "medium",
        content_type: str = "other",
        sensitive_regions: list = None,
        **kwargs                    # ✅ 吸收未知字段，防止报错
):
    """
    在 JPG/JPEG 图像的 DCT 域嵌入 UID。

    Args:
        strength: LLM 决策的嵌入强度 ("low"/"medium"/"high")
        content_type: LLM 识别的内容类型（用于日志/审计）
        sensitive_regions: LLM 识别的敏感区域坐标列表（v2.4 预留）
    """
    # ✅ 根据 LLM 策略动态设置嵌入强度
    threshold = STRENGTH_MAP.get(strength, DEFAULT_THRESHOLD)
    print(f"[DCT] 嵌入参数: strength={strength}(threshold={threshold}), content_type={content_type}")

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {input_path}")

    h_orig, w_orig = img_bgr.shape[:2]
    h_pad = (8 - h_orig % 8) % 8
    w_pad = (8 - w_orig % 8) % 8
    img_bgr = np.pad(img_bgr, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant')

    h, w = img_bgr.shape[:2]
    blocks_r, blocks_g, blocks_b = [], [], []

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img_bgr[i:i+8, j:j+8]
            blocks_r.append(block[:, :, 2].astype(np.float32))
            blocks_g.append(block[:, :, 1].astype(np.float32))
            blocks_b.append(block[:, :, 0].astype(np.float32))

    binary_str = _encode_uid_to_bits(uid)
    bits = [int(b) for b in binary_str]
    row, col = DEFAULT_EMBED_POS

    for idx, bit in enumerate(bits):
        if idx >= len(blocks_r):
            break
        block_r = blocks_r[idx]
        dct_block = dct(dct(block_r.T, norm='ortho').T, norm='ortho')

        # ✅ 使用动态 threshold 替代硬编码
        if bit == 1:
            dct_block[row, col] += threshold
        else:
            dct_block[row, col] -= threshold

        idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        blocks_r[idx] = np.clip(idct_block, 0, 255).astype(np.uint8)

    new_img = np.zeros((h, w, 3), dtype=np.uint8)
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if idx < len(blocks_r):
                new_img[i:i+8, j:j+8, 2] = blocks_r[idx]
                new_img[i:i+8, j:j+8, 1] = blocks_g[idx]
                new_img[i:i+8, j:j+8, 0] = blocks_b[idx]
                idx += 1

    cv2.imwrite(output_path, new_img)


def extract_uid_from_image(img_path: str) -> str:
    """提取逻辑不变（提取不需要知道嵌入强度）"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")

    h, w = img_bgr.shape[:2]
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    img_bgr = np.pad(img_bgr, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant')

    img_r = img_bgr[:, :, 2].astype(np.float32)
    blocks = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blocks.append(img_r[i:i+8, j:j+8])

    watermark_bits = ""
    row, col = DEFAULT_EMBED_POS
    max_bits = 10000

    for idx in range(min(len(blocks), max_bits)):
        dct_block = dct(dct(blocks[idx].T, norm='ortho').T, norm='ortho')
        watermark_bits += '1' if dct_block[row, col] > 0 else '0'

    try:
        return _decode_bits_to_uid(watermark_bits)
    except Exception as e:
        raise ValueError("无效的 DCT 水印数据") from e