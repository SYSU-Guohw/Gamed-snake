from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import numpy as np

from lib.networks.darnet.drn import BasicBlock
from lib.networks.darnet import drn



class UpConvBlock(nn.Module):  # 见forward函数里的注释。
    def __init__(self, inplanes, planes):
        super(UpConvBlock, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(inplanes, planes, 4, stride=2, padding=1, output_padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, 4, stride=2, padding=1, bias=False),  # stride=2: 卷积的步长，设置为2意味着每次卷积运算会将特征图的每个维度的尺寸扩大2倍
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = x
        x = self.upconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        residual = self.upsample(residual)
        x = self.relu(x + residual)
        # 从这个forward看来，就是先反卷积一次，BN、RELU，然后再卷积一次（没变分辨率），
        # 最后再反卷积，BN，残差连接输出。这样看来，应该是反卷积了2次。
        return x


class DRNContours(nn.Module):
    def __init__(self, model_name='drn_d_22', classes=3, pretrained=True):  # classes = 3  :  预测的特征图3个通道
        super(DRNContours, self).__init__()
        model = drn.__dict__.get(model_name)(pretrained=pretrained, num_classes=1000)
        # print(model_name)
        # 上句，可能是根据那个model_name（如果默认值就是drn_d_22），
        #     从drn.py中的drn_a_50/drn_c_26/drn_c_42/drn_c_58……等等中选出来一个网络。
        # 这应该是用来构建下面的那个self.base用的，也就是说，用了上面drn_c_26的、除了最后两层之外的其他层。
        self.base = nn.Sequential(*list(model.children())[:-2])  # 这种取模型部分模块的方法很不错，值得学习
        # 上句，可能是把模型中除了最后两层中的卷积层什么的，都给他拿出来。从后面forward函数里看，是作为基础特征提取器用的。
        self.upconv1 = UpConvBlock(model.out_dim, model.out_dim // 2)
        self.upconv2 = UpConvBlock(model.out_dim // 2, model.out_dim // 4)
        self.upconv3 = UpConvBlock(model.out_dim // 4, model.out_dim // 8)  # 在上采样过程中，特征图尺寸不断增大，但通道数不断压缩（一次减少一半）
        # 以上，定义上采样层。从class UpConvBlock的forward函数看，上面self.upconv1/2/3都是做了2次上采样。
        #     然后从后面的forward函数里看，就是用self.base得到的特征之后，串行弄这个self.upconv1/2/3，接连做了2*3=6次上采样。
        self.predict_1 = nn.Conv2d(model.out_dim // 8, classes, kernel_size=3, stride=1, padding=1, bias=True)
        # 上句，预测特征图（看起来是3个通道的），
        #     从forward函数看，就是对应蛇算法里的A图（一阶内力）、B图（二阶内力）和K图（气球力能量），
        #     而E图（图像能量）则不是self.predict_1的输出，估计是用前面那个x弄的。
        self.conv4 = self._create_convblock(model.out_dim // 8 + classes, model.out_dim // 16, 1)
        self.conv5 = self._create_convblock(model.out_dim // 16, model.out_dim // 32, 2)
        self.conv6 = self._create_convblock(model.out_dim // 32, model.out_dim // 32, 1)
        # 以上conv4~cpnv6，写的是Combine predictions with output of upconv 3 to further refine，
        #     从后面forward函数看，是把那些能量图（E、A、B、K图）拼起来，再经过了self.conv4/5/6做卷积，然后下采样，
        #     把蛇算法里的图又预测了一次。
        self.predict_2 = nn.Conv2d(model.out_dim // 32, classes, kernel_size=3, stride=1, padding=1, bias=True)
        # 上句，经过了self.conv4/5/6做卷积之后，重新预测蛇算法对应的图。

        modules_to_init = [self.upconv1, self.upconv2, self.upconv3, self.predict_1,
                           self.conv4, self.conv5, self.conv6]
        for m in modules_to_init:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # 各种初始化。
    
    def _create_convblock(self, in_planes, out_planes, dilation):
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        convblock = BasicBlock(in_planes, out_planes, dilation=(dilation, dilation), downsample=downsample)
        # 上面，设计一个带下采样的卷积块。就是调用BasicBlock类，只不过这次传入一个“downsample”作为下采样方法，
        #     从那个BasicBlock分forward函数，可以看出来convblock就是个有downsample模块的BasicBlock，
        #     也就是说，比起一般的BasicBlock多下采样了一次。
        return convblock

    def forward(self, x):
        x = self.base(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        pr1 = self.predict_1(x)
        pr1 = nn.functional.relu(pr1)  # pr1为第一次预测出来的能量图
        beta1 = pr1[:, 0, :, :]
        data1 = pr1[:, 1, :, :]
        kappa1 = pr1[:, 2, :, :]
        x = self.conv4(torch.cat((x, pr1), dim=1))   # 这里将x和pr1拼接后继续进行卷积（进行了一个残差操作）
        x = self.conv5(x)
        x = self.conv6(x)
        pr2 = self.predict_2(x)
        pr2 = nn.functional.relu(pr2)
        beta2 = pr2[:, 0, :, :]
        data2 = pr2[:, 1, :, :]
        kappa2 = pr2[:, 2, :, :]
        # 见上面__init__的注释，已经分析明白网络结构了。
        #     总之是预测了两次蛇算法的E、A、B、K图，网络相较于014文章要复杂一些，很多上下采样结构。
        return beta1, data1, kappa1, beta2, data2, kappa2


