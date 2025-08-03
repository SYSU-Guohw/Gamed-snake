import torch.nn as nn
from .dla import DLASeg, dla34, DLAUp, IDAUp  # 添加 DLAUp 和 IDAUp 的导入
from .evolve import Evolution
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode
import torch
from lib.config import cfg
import warnings
import numpy as np
import logging
from lib.networks.classifier.wave_mlp import WaveMLP

warnings.filterwarnings("ignore")

class MultiModalFusion(nn.Module):
    def __init__(self, text_dim=768, image_dim=64):
        super(MultiModalFusion, self).__init__()
        
        # 文本特征投影层
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU(inplace=True)
        )
        
        # 空间自适应模块
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(image_dim, image_dim, 1),
            nn.BatchNorm2d(image_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力融合
        self.attention = nn.Sequential(
            nn.Conv2d(image_dim * 2, image_dim, 1),
            nn.BatchNorm2d(image_dim),
            nn.Sigmoid()
        )
        
        # 残差输出投影
        self.residual_proj = nn.Sequential(
            nn.Conv2d(image_dim, image_dim, 1),
            nn.BatchNorm2d(image_dim)
        )

    def forward(self, img_feat, text_feat):
        B, C, H, W = img_feat.shape
        
        # 1. 投影文本特征
        text_feat = self.text_proj(text_feat)  # [B, image_dim]
            
        # 2. 扩展到空间维度
        text_feat = text_feat.view(B, -1, 1, 1)  # [B, image_dim, 1, 1]
        text_feat = text_feat.expand(-1, -1, H, W)  # [B, image_dim, H, W]
        
        # 3. 空间自适应
        text_feat = self.spatial_proj(text_feat)
        
        # 4. 计算注意力权重
        fusion_feat = torch.cat([img_feat, text_feat], dim=1)  
        attention_weights = self.attention(fusion_feat)
        
        # 5. 计算残差
        residual = self.residual_proj(text_feat * attention_weights)
        
        # 6. 残差连接 - 保持原始特征不变
        output = img_feat + residual
        
        return output

class GlobalTextFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x, text_feat):
        B, C, H, W = x.shape
        
        # 扩展文本特征到空间维度
        text_feat = text_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)
        text_feat = self.spatial_proj(text_feat)
        
        # 计算通道注意力
        channel_weight = self.channel_attention(text_feat)
        
        # 计算空间注意力
        avg_feat = torch.mean(text_feat, dim=1, keepdim=True)
        max_feat, _ = torch.max(text_feat, dim=1, keepdim=True)
        spatial_weight = self.spatial_attention(torch.cat([avg_feat, max_feat], dim=1))
        
        # 应用注意力
        enhanced = x * channel_weight * spatial_weight
        
        return enhanced

def fill_fc_weights(layers):
    """初始化全连接层的权重"""
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                           [2 ** i for i in range(self.last_level - self.first_level)])
        
        # 添加文本编码器
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        # self.text_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        
        # 文本特征融合模块
        self.multimodal_fusion = MultiModalFusion(text_dim=768, image_dim=channels[self.first_level])
        
        # 门控参数
        self.fusion_gate = nn.Parameter(torch.zeros(1))
        
        # 初始化其他头部网络
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(out_channel, head_conv,
                             kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                             kernel_size=final_kernel, stride=1,
                             padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(out_channel, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

        # 双分类头部分
        self.class_head = WaveMLP('M', num_classes=cfg.heads['ct_hm'])

    def forward(self, x):
        # 获取图像特征
        y = []
        x = self.base(x)
        x = self.dla_up(x)
        
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return z, y[-1]

class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv)
        self.gcn = Evolution()

    def forward(self, x, batch=None):
        # DLA 网络处理
        output, cnn_feature = self.dla(x)
        
        # 解码检测结果
        ct, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'detection': detection})
        output.update({'ct': ct})

        if cfg.use_gt_det:
            self.use_gt_detection(output, batch)

        # GCN 处理
        output = self.gcn(output, cnn_feature, batch)
        
        return output

    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection

    def use_gt_detection(self, output, batch):
        _, _, height, width = output['ct_hm'].size()
        ct_01 = batch['ct_01'].byte()

        ct_ind = batch['ct_ind'][ct_01]
        xs, ys = ct_ind % width, ct_ind // width
        xs, ys = xs[:, None].float(), ys[:, None].float()
        ct = torch.cat([xs, ys], dim=1)

        wh = batch['wh'][ct_01]
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=1)
        score = torch.ones([len(bboxes)]).to(bboxes)[:, None]
        ct_cls = batch['ct_cls'][ct_01].float()[:, None]
        detection = torch.cat([bboxes, score, ct_cls], dim=1)

        output['ct'] = ct[None]
        output['detection'] = detection[None]

        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network


