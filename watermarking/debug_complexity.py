# 调试复杂度计算
import cv2
import numpy as np

def analyze_image_complexity_debug(image_path: str) -> dict:
    """调试版本的复杂度分析"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return {}

    height, width = img.shape

    # 方法1：边缘密度
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges) / (height * width)
    edge_score = min(1.0, edge_density * 10)  # 调整归一化参数

    # 方法2：纹理方差
    gray = cv2.GaussianBlur(img, (5, 5), 0)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(laplacian)
    texture_score = min(1.0, texture_variance / 500)  # 调整归一化参数

    # 方法3：颜色方差
    color_std = np.std(img)
    color_score = min(1.0, color_std / 50)  # 调整归一化参数

    # 方法4：直方图熵（纹理复杂度）
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # 去除0
    entropy = -np.sum(hist * np.log2(hist))
    entropy_score = min(1.0, entropy / 7)  # 熵通常在5-7之间

    # 综合复杂度（调整权重）
    complexity = (edge_score * 0.25 + texture_score * 0.25 + color_score * 0.25 + entropy_score * 0.25)

    return {
        "edge_density": edge_density,
        "edge_score": edge_score,
        "texture_variance": texture_variance,
        "texture_score": texture_score,
        "color_std": color_std,
        "color_score": color_score,
        "entropy": entropy,
        "entropy_score": entropy_score,
        "complexity": complexity
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python debug_complexity.py <image_path>")
        print("示例: python debug_complexity.py test.png")
        sys.exit(1)

    image_path = sys.argv[1]
    result = analyze_image_complexity_debug(image_path)

    print(f"\n=== 图片: {image_path} ===")
    print(f"\n边缘密度: {result['edge_density']:.6f}")
    print(f"边缘得分: {result['edge_score']:.3f}")
    print(f"\n纹理方差: {result['texture_variance']:.2f}")
    print(f"纹理得分: {result['texture_score']:.3f}")
    print(f"\n颜色标准差: {result['color_std']:.2f}")
    print(f"颜色得分: {result['color_score']:.3f}")
    print(f"\n直方图熵: {result['entropy']:.3f}")
    print(f"熵得分: {result['entropy_score']:.3f}")
    print(f"\n综合复杂度: {result['complexity']:.3f}")

    # 根据复杂度判断强度
    if result['complexity'] > 0.6:
        strength = 0.8
        label = "高"
    elif result['complexity'] > 0.4:
        strength = 0.6
        label = "中"
    elif result['complexity'] > 0.2:
        strength = 0.4
        label = "中低"
    else:
        strength = 0.3
        label = "低"

    print(f"\n建议强度: {strength*100:.0f}% ({label})")
