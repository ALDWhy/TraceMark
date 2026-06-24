# adaptive_watermark.py - 水印强度自适应模块
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
import os

# 强度级别定义
STRENGTH_LEVELS = {
    "low": {"name": "低强度", "description": "适合平滑区域，保护图像质量"},
    "medium": {"name": "中等强度", "description": "平衡安全性和质量"},
    "high": {"name": "高强度", "description": "适合复杂区域，增强鲁棒性"},
    "auto": {"name": "自适应", "description": "根据图像内容自动调整"}
}


def analyze_image_complexity(image_path: str) -> float:
    """
    分析图像复杂度（0-1，越高越复杂）
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        复杂度分数（0-1）
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.5  # 默认中等复杂度
    
    # 方法1：计算边缘密度（使用 Canny 边缘检测）
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])
    # 调整归一化：边缘密度通常在5-30之间，30算非常复杂
    edge_score = min(1.0, edge_density / 30)
    
    # 方法2：计算纹理方差（Laplacian）
    gray = cv2.GaussianBlur(img, (5, 5), 0)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(laplacian)
    # 调整归一化：纹理方差通常在10-100之间，100算非常复杂
    texture_score = min(1.0, texture_variance / 100)
    
    # 方法3：计算颜色方差（颜色越多越复杂）
    color_std = np.std(img)
    # 调整归一化：颜色标准差通常在20-80之间，80算非常复杂
    color_score = min(1.0, color_std / 80)
    
    # 方法4：直方图熵（纹理复杂度）
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    # 熵通常在5-8之间，8算非常复杂
    entropy_score = min(1.0, entropy / 8)
    
    # 综合复杂度指标（加权平均）
    complexity = (edge_score * 0.25 + texture_score * 0.25 + color_score * 0.25 + entropy_score * 0.25)
    
    return min(1.0, max(0.0, complexity))


def detect_sensitive_regions(image_path: str) -> Dict[str, Any]:
    """
    检测图像中的敏感区域（主体、文字等）
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        敏感区域检测结果
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"has_subject": False, "subject_type": "none", "has_text": False, "sensitive_ratio": 0.0}
    
    height, width = img.shape[:2]
    sensitive_pixels = 0
    
    # 使用 Haar 分类器检测主体（人脸或动物面部）
    # 使用多个分类器提高检测率
    cascade_paths = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_alt_tree.xml'
    ]
    
    detected_regions = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for cascade_name in cascade_paths:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
        if face_cascade.empty():
            continue
        
        regions = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.15,    # 适度的缩放因子
            minNeighbors=4,      # 适度的邻居验证
            minSize=(30, 30),    # 较小的最小尺寸，适应动物面部
            maxSize=(int(width*0.8), int(height*0.8))  # 限制最大尺寸
        )
        
        # 合并检测结果，去重
        for (x, y, w, h) in regions:
            # 检查是否与已有区域重叠
            is_new = True
            for (x2, y2, w2, h2) in detected_regions:
                if abs(x - x2) < 20 and abs(y - y2) < 20:
                    is_new = False
                    break
            if is_new:
                detected_regions.append((x, y, w, h))
    
    has_subject = len(detected_regions) > 0
    subject_type = "subject"  # 使用更通用的术语
    confidence = "low"
    
    # 根据检测到的区域大小判断置信度
    if has_subject:
        total_area = 0
        for (x, y, w, h) in detected_regions:
            region_area = w * h
            total_area += region_area
            sensitive_pixels += region_area
        
        # 根据区域占图像的比例判断置信度
        region_ratio = total_area / (height * width)
        if region_ratio > 0.05:
            confidence = "high"
        elif region_ratio > 0.02:
            confidence = "medium"
    
    # 文字区域检测（极其保守，避免误判）
    has_text = False
    text_confidence = 0
    
    # 方法：检测密集的水平和垂直线条模式（只有真正的文档才会有）
    # 使用非常严格的阈值
    edges = cv2.Canny(gray, 150, 250)  # 更高的边缘阈值
    
    # 霍夫变换检测直线（更严格的参数）
    lines = cv2.HoughLinesP(
        edges, 
        1, 
        np.pi/180, 
        threshold=50,      # 更高的阈值
        minLineLength=30,  # 更长的线段
        maxLineGap=3       # 更小的间隙
    )
    
    if lines is not None and len(lines) > 50:  # 要求至少50条直线
        # 统计水平和垂直直线的数量
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 5 or angle > 175:
                horizontal_lines += 1
            elif 85 < angle < 95:
                vertical_lines += 1
        
        # 要求大量的水平和垂直直线（只有文档才会有）
        # 且比例必须接近（排除栅栏等）
        if horizontal_lines > 40 and vertical_lines > 40:
            ratio = horizontal_lines / max(1, vertical_lines)
            if 0.5 < ratio < 2.0:
                # 进一步检查：这些直线是否形成网格状排列
                has_text = True
                text_confidence = 0.7
                sensitive_pixels += int(height * width * 0.05)
    
    sensitive_ratio = min(1.0, sensitive_pixels / (height * width))
    
    return {
        "has_subject": has_subject,
        "subject_type": subject_type,
        "confidence": confidence,
        "has_text": has_text,
        "sensitive_ratio": round(sensitive_ratio, 2),
        "regions_count": len(detected_regions)
    }


def calculate_adaptive_strength(image_path: str) -> Dict[str, Any]:
    """
    根据图像内容计算最佳水印强度
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        自适应强度计算结果
    """
    # 分析图像复杂度
    complexity = analyze_image_complexity(image_path)
    
    # 检测敏感区域
    sensitive_info = detect_sensitive_regions(image_path)
    
    # 根据复杂度计算基础强度（调整阈值，让不同图片有明显区别）
    if complexity > 0.7:
        base_strength = 0.8  # 高强度（复杂图像）
        strength_label = "high"
    elif complexity > 0.5:
        base_strength = 0.6  # 中等强度
        strength_label = "medium"
    elif complexity > 0.3:
        base_strength = 0.4  # 中低强度
        strength_label = "low"
    else:
        base_strength = 0.3  # 低强度（简单图像）
        strength_label = "low"
    
    # 根据敏感区域调整强度（降低惩罚力度）
    adjusted_strength = base_strength * (1 - sensitive_info["sensitive_ratio"] * 0.3)
    
    # 确保强度在合理范围内（提高下限）
    adjusted_strength = max(0.25, min(0.9, adjusted_strength))
    
    return {
        "complexity": round(complexity, 2),
        "complexity_level": "高" if complexity > 0.7 else ("中" if complexity > 0.4 else "低"),
        "sensitive_info": sensitive_info,
        "base_strength": round(base_strength, 2),
        "adjusted_strength": round(adjusted_strength, 2),
        "strength_label": strength_label,
        "strategy": {
            "bits_per_pixel": int(round(adjusted_strength * 3)) + 1,  # 1-4 bits
            "dct_coefficient": int(round(adjusted_strength * 50)) + 10,
            "description": STRENGTH_LEVELS[strength_label]["description"]
        }
    }


def embed_adaptive_watermark(image_path: str, output_path: str, uid: str) -> Dict[str, Any]:
    """
    使用自适应强度嵌入水印
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        uid: 水印 UID
        
    Returns:
        嵌入结果和分析数据
    """
    # 计算自适应强度
    analysis = calculate_adaptive_strength(image_path)
    strength = analysis["adjusted_strength"]
    
    # 根据图像格式选择嵌入算法
    ext = os.path.splitext(image_path)[1].lower()
    
    # 根据强度调整嵌入参数
    embed_kwargs = {
        "strength": strength,
        "bits_per_pixel": analysis["strategy"]["bits_per_pixel"]
    }
    
    # 执行嵌入（调用现有的 LSB 或 DCT 嵌入函数）
    try:
        if ext == '.png':
            # LSB 嵌入
            from .lsb import embed_uid_in_image
            embed_uid_in_image(image_path, output_path, uid, **embed_kwargs)
        else:
            # DCT 嵌入
            from .dct import embed_uid_in_image
            embed_uid_in_image(image_path, output_path, uid, **embed_kwargs)
        
        return {
            "status": "success",
            "uid": uid,
            "analysis": analysis,
            "output_path": output_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "analysis": analysis
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python adaptive_watermark.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=== 水印强度自适应分析 ===")
    result = calculate_adaptive_strength(image_path)
    
    print(f"图像复杂度: {result['complexity']} ({result['complexity_level']})")
    print(f"敏感区域检测:")
    print(f"  - 含人脸: {'是' if result['sensitive_info']['has_face'] else '否'}")
    print(f"  - 含文字: {'是' if result['sensitive_info']['has_text'] else '否'}")
    print(f"  - 敏感区域比例: {result['sensitive_info']['sensitive_ratio']}%")
    print(f"基础强度: {result['base_strength']}")
    print(f"调整后强度: {result['adjusted_strength']}")
    print(f"强度级别: {STRENGTH_LEVELS[result['strength_label']]['name']}")
    print(f"嵌入策略:")
    print(f"  - 每像素嵌入位数: {result['strategy']['bits_per_pixel']}")
    print(f"  - DCT 系数: {result['strategy']['dct_coefficient']}")