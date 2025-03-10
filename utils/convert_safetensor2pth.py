import sys
import os
import torch
from safetensors.torch import load_file

# 将父目录添加到系统路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

try:
    # 尝试导入 DBANet 模块
    from models.DBANet import DBANet
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print(f"当前系统路径: {sys.path}")

# 加载safetensor文件
safetensors_path = '/home/xxx/Aproj/models/backbone/model.safetensors'  # 替换为你的safetensor文件路径
state_dict = load_file(safetensors_path)

# 创建模型实例
model = DBANet()

# 只加载匹配的权重
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(pretrained_dict)

# 加载匹配的权重
model.load_state_dict(model_dict, strict=False) 

# 将模型保存为pth文件
pth_path = '/home/xxx/Aproj/models/backbone/pretrained/pvt_v2_b5.pth'  # 替换为你想要保存的pth文件路径
torch.save(model.state_dict(), pth_path)