import os
import torch
from .u2net import U2NET

# 获取当前脚本所在目录（即 watermarking/）
current_dir = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(current_dir, "u2net.pth")

print(f"Loading weights from: {weight_path}")
state_dict = torch.load(weight_path, map_location="cpu")
model = U2NET()
model.load_state_dict(state_dict)
model.eval()
print("✅ 模型权重加载成功！")