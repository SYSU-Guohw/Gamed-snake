import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CentralDifferentialConvolution(nn.Module):
    """中央差分卷积 (CDC) - 计算中心元素与相邻元素的差值"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CentralDifferentialConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 创建可学习的卷积核
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
    def forward(self, x):
        # 计算中央差分
        # 获取输入尺寸
        batch_size, channels, height, width = x.size()
        
        # 创建差分特征图
        diff_features = torch.zeros_like(x)
        
        # 计算中央差分：中心元素与相邻元素的差值
        # 水平方向差分
        diff_features[:, :, :, 1:] += x[:, :, :, :-1] - x[:, :, :, 1:]  # 右减左
        diff_features[:, :, :, :-1] += x[:, :, :, 1:] - x[:, :, :, :-1]  # 左减右
        
        # 垂直方向差分
        diff_features[:, :, 1:, :] += x[:, :, :-1, :] - x[:, :, 1:, :]  # 下减上
        diff_features[:, :, :-1, :] += x[:, :, 1:, :] - x[:, :, :-1, :]  # 上减下
        
        # 对角线差分
        diff_features[:, :, 1:, 1:] += x[:, :, :-1, :-1] - x[:, :, 1:, 1:]  # 右下减左上
        diff_features[:, :, :-1, :-1] += x[:, :, 1:, 1:] - x[:, :, :-1, :-1]  # 左上减右下
        
        diff_features[:, :, 1:, :-1] += x[:, :, :-1, 1:] - x[:, :, 1:, :-1]  # 左下减右上
        diff_features[:, :, :-1, 1:] += x[:, :, 1:, :-1] - x[:, :, :-1, 1:]  # 右上减左下
        
        # 应用卷积
        output = self.conv(diff_features)
        return output


class DiagonalDifferentialConvolution(nn.Module):
    """对角线差分卷积 (DDC) - 计算对角线方向的元素差值"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DiagonalDifferentialConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 创建可学习的卷积核
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
    def forward(self, x):
        # 计算对角线差分
        batch_size, channels, height, width = x.size()
        
        # 创建差分特征图
        diff_features = torch.zeros_like(x)
        
        # 主对角线差分 (左上到右下)
        diff_features[:, :, 1:, 1:] += x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
        diff_features[:, :, :-1, :-1] += x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        
        # 副对角线差分 (右上到左下)
        diff_features[:, :, 1:, :-1] += x[:, :, :-1, 1:] - x[:, :, 1:, :-1]
        diff_features[:, :, :-1, 1:] += x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        
        # 应用卷积
        output = self.conv(diff_features)
        return output


class SpatialDifferentialConvolution(nn.Module):
    """空间差分卷积 (SDC) - 计算特定空间方向的元素差值"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SpatialDifferentialConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 创建可学习的卷积核
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
    def forward(self, x):
        # 计算空间差分
        batch_size, channels, height, width = x.size()
        
        # 创建差分特征图
        diff_features = torch.zeros_like(x)
        
        # 计算8个方向的差分
        # 上
        if height > 1:
            diff_features[:, :, 1:, :] += x[:, :, :-1, :] - x[:, :, 1:, :]
        # 下
        if height > 1:
            diff_features[:, :, :-1, :] += x[:, :, 1:, :] - x[:, :, :-1, :]
        # 左
        if width > 1:
            diff_features[:, :, :, 1:] += x[:, :, :, :-1] - x[:, :, :, 1:]
        # 右
        if width > 1:
            diff_features[:, :, :, :-1] += x[:, :, :, 1:] - x[:, :, :, :-1]
        # 左上
        if height > 1 and width > 1:
            diff_features[:, :, 1:, 1:] += x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
        # 右上
        if height > 1 and width > 1:
            diff_features[:, :, 1:, :-1] += x[:, :, :-1, 1:] - x[:, :, 1:, :-1]
        # 左下
        if height > 1 and width > 1:
            diff_features[:, :, :-1, 1:] += x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        # 右下
        if height > 1 and width > 1:
            diff_features[:, :, :-1, :-1] += x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        
        # 应用卷积
        output = self.conv(diff_features)
        return output


class DCIM(nn.Module):
    """差分卷积信息模块 (Differential Convolutional Information Module)"""
    
    def __init__(self, in_channels, out_channels, reduction=4):
        super(DCIM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 计算每个分支的输出通道数
        branch_channels = out_channels // 4
        
        # CDC分支
        self.cdc_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            CentralDifferentialConvolution(branch_channels, branch_channels, 3, 1, 1)
        )
        
        # DDC分支
        self.ddc_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            DiagonalDifferentialConvolution(branch_channels, branch_channels, 3, 1, 1)
        )
        
        # SDC分支
        self.sdc_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            SpatialDifferentialConvolution(branch_channels, branch_channels, 3, 1, 1)
        )
        
        # 池化分支
        self.pool_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 输出融合层
        self.fusion = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # CDC分支
        cdc_out = self.cdc_branch(x)
        
        # DDC分支
        ddc_out = self.ddc_branch(x)
        
        # SDC分支
        sdc_out = self.sdc_branch(x)
        
        # 池化分支
        pool_out = self.pool_branch(x)
        pool_out = pool_out.expand(-1, -1, height, width)  # 扩展到原始尺寸
        
        # 特征拼接
        concat_features = torch.cat([cdc_out, ddc_out, sdc_out, pool_out], dim=1)
        
        # 融合输出
        output = self.fusion(concat_features)
        output = self.bn(output)
        output = self.relu(output)
        
        return output


class EnergyGradientExtractor(nn.Module):
    """能量梯度提取器 - 使用DCIM模块进行能量图梯度提取"""
    
    def __init__(self, in_channels=64, out_channels=64):
        super(EnergyGradientExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 左侧路径：传统卷积
        self.left_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 中心路径：DCIM模块
        self.center_path = nn.Sequential(
            DCIM(in_channels, out_channels),
            DCIM(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 右侧路径：简单卷积
        self.right_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 输出融合
        self.output_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 左侧路径
        left_out = self.left_path(x)
        
        # 中心路径
        center_out = self.center_path(x)
        
        # 右侧路径
        right_out = self.right_path(x)
        
        # 特征融合
        concat_features = torch.cat([left_out, center_out, right_out], dim=1)
        output = self.output_fusion(concat_features)
        
        return output 