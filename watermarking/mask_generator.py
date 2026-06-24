# watermarking/mask_generator.py
import cv2
import numpy as np
import torch
from u2net import U2NET

class SemanticMaskGenerator:
    def __init__(self, model_path='u2net.pth'):
        self.model = U2NET()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def generate(self, image_np: np.ndarray, strategy: dict) -> np.ndarray:
        h, w = image_np.shape[:2]
        input_tensor = self._preprocess(image_np)

        with torch.no_grad():
            d1 = self.model(input_tensor)[0]

        sal_map = d1.squeeze().cpu().numpy()
        sal_map = cv2.resize(sal_map, (w, h), interpolation=cv2.INTER_LINEAR)
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)

        # 根据 LLM 策略动态调整阈值
        strength = strategy.get("embed_strength", "medium")
        threshold_map = {"high": 0.2, "medium": 0.4, "low": 0.6}
        thresh = threshold_map.get(strength, 0.4)
        mask = (sal_map > thresh).astype(np.float32)

        # 敏感区域形态学保护
        sensitive = strategy.get("sensitive_regions", [])
        if "face" in sensitive or "text" in sensitive:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    def _preprocess(self, img):
        img = cv2.resize(img, (1024, 1024)).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)