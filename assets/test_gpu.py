import torch
import torch.nn as nn
from torch.nn import DataParallel

# 假设你的模型定义如下
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
network = MyNetwork()

# 将模型迁移到 GPU
network = network.cuda()  # 将模型迁移到默认的 GPU 设备

# 使用 DataParallel 支持多 GPU
network = DataParallel(network)

# 查看当前默认的 GPU 设备
current_device = torch.cuda.current_device()
print(f"当前默认的 GPU 设备索引: {current_device}")
print(f"当前默认的 GPU 设备名称: {torch.cuda.get_device_name(current_device)}")

# 查看所有可用的 GPU 设备
num_gpus = torch.cuda.device_count()
print(f"可用的 GPU 数量: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 查看模型参数所在的设备
for param in network.parameters():
    print(f"模型参数所在的设备: {param.device}")
    break  # 只需检查一个参数即可

# 查看 DataParallel 使用的设备
print(f"DataParallel 使用的设备: {network.device_ids}")