import torch
import torch.nn as nn
from lib.networks.vision_mamba2.mamba2 import VMAMBA2Block
# 假设 VMAMBA2Block 已经定义好了，且其他代码完整且正确

# # 定义输入参数
dim = 128  # 输入通道数
input_resolution = 66  # 输入分辨率
# num_heads = 8  # 注意力头数
# mlp_ratio = 4.0  # MLP隐藏层维度与嵌入维度的比例
# qkv_bias = True  # 是否为QKV添加可学习的偏置
# drop = 0.0  # Dropout率
# drop_path = 0.0  # 随机深度率
# act_layer = nn.GELU  # 激活层
# norm_layer = nn.LayerNorm  # 归一化层
# ssd_expansion = 2  # SSD扩展因子
# ssd_ngroups = 1  # SSD分组数
# ssd_chunk_size = 256  # SSD块大小
# linear_attn_duality = True  # 是否使用线性注意力双重性
# d_state = 64  # 状态维度

# 创建 VMAMBA2Block 实例
block = VMAMBA2Block(
    dim=128,
    # input_resolution=input_resolution, # 序列长度
    # num_heads=num_heads,
    # mlp_ratio=mlp_ratio,
    # qkv_bias=qkv_bias,
    # drop=drop,
    # drop_path=drop_path,
    # act_layer=act_layer,
    # norm_layer=norm_layer,
    # ssd_expansion=ssd_expansion,
    # ssd_ngroups=ssd_ngroups,
    # ssd_chunk_size=ssd_chunk_size,
    # linear_attn_duality=linear_attn_duality,
    # d_state=d_state,
)

# 创建随机输入张量
batch_size = 2  # 批量大小
seq_length = input_resolution  # 序列长度
input_tensor = torch.randn(batch_size, seq_length, dim)  # (B, L, C)
print("Iutput shape:", input_tensor.shape)

# 前向传播
output = block(input_tensor)

# 打印输出形状
print("Output shape:", output.shape)