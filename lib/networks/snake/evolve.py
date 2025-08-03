import torch.nn as nn
from .snake import Snake
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode, active_spline
import torch
from lib.networks.vision_mamba2.mamba2 import VMAMBA2Block
import warnings
from lib.config import cfg
import os
import torch.nn.functional as F
import math
# 设置tokenizer并行化选项
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # 图像特征和文本特征的投影
        self.img_q = nn.Linear(dim, dim)
        self.img_k = nn.Linear(dim, dim)
        self.img_v = nn.Linear(dim, dim)
        
        self.txt_q = nn.Linear(dim, dim)
        self.txt_k = nn.Linear(dim, dim)
        self.txt_v = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim * 2, dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        
        # 缩放因子
        self.scale = dim ** -0.5
        
    def forward(self, img_feat, txt_feat):
        """
        img_feat: [N, L1, D]
        txt_feat: [N, L2, D] 
        注意：L1和L2可以不相等
        """
        # 保存残差连接
        residual = img_feat
        
        # 投影特征
        img_q = self.img_q(img_feat)  # [N, L1, D]
        img_k = self.img_k(img_feat)  # [N, L1, D]
        img_v = self.img_v(img_feat)  # [N, L1, D]
        
        txt_q = self.txt_q(txt_feat)  # [N, L2, D]
        txt_k = self.txt_k(txt_feat)  # [N, L2, D]
        txt_v = self.txt_v(txt_feat)  # [N, L2, D]
        
        # 计算注意力 (调整维度以适应不同序列长度)
        attn_img2txt = torch.matmul(img_q, txt_k.transpose(-2, -1)) * self.scale  # [N, L1, L2]
        attn_img2txt = F.softmax(attn_img2txt, dim=-1)
        img2txt_feat = torch.matmul(attn_img2txt, txt_v)  # [N, L1, D]
        
        attn_txt2img = torch.matmul(txt_q, img_k.transpose(-2, -1)) * self.scale  # [N, L2, L1]
        attn_txt2img = F.softmax(attn_txt2img, dim=-1)
        txt2img_feat = torch.matmul(attn_txt2img, img_v)  # [N, L2, D]
        
        # 对齐序列长度 (使用平均池化)
        txt2img_feat = F.adaptive_avg_pool1d(
            txt2img_feat.transpose(1, 2), 
            img_feat.size(1)
        ).transpose(1, 2)  # [N, L1, D]
        
        # 特征融合
        fused_feat = torch.cat([img2txt_feat, txt2img_feat], dim=-1)  # [N, L1, 2D]
        fused_feat = self.out_proj(fused_feat)  # [N, L1, D]
        
        # 添加残差连接和层归一化
        fused_feat = self.layer_norm1(residual + fused_feat)
        
        return self.layer_norm2(fused_feat)


class MultiModalFusion(nn.Module):
    def __init__(self, text_dim=768, feat_dim=64, num_points=128):
        super().__init__()
        self.text_dim = text_dim
        self.feat_dim = feat_dim
        self.num_points = num_points
        
        # 1. 文本特征处理
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, feat_dim * 4),
            nn.LayerNorm(feat_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 4, feat_dim)
        )
        
        # 2. 特征转换模块
        self.transform = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, feat_dim)
        )
        
        # 3. 门控机制
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )
        
        # 4. 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # 5. 输出层归一化
        self.norm = nn.LayerNorm(feat_dim)
        
    def forward(self, contour_feat, text_feat):
        """
        contour_feat: [B, num_points, feat_dim] - 轮廓特征，B为轮廓数量
        text_feat: [N, text_dim] - 文本特征，N为类别数量
        """
        B, L, D = contour_feat.size()
        N = text_feat.size(0)
        
        # 1. 先进行文本特征处理
        projected_text = self.text_projection(text_feat)  # [N, feat_dim]
        
        # 2. 记录每个轮廓对应的类别索引 (使用最近邻匹配)
        text_emb = F.normalize(projected_text, dim=-1)
        contour_emb = F.normalize(contour_feat.mean(dim=1), dim=-1)  # [B, feat_dim]
        sim_matrix = torch.matmul(contour_emb, text_emb.t())  # [B, N]
        matched_idx = sim_matrix.max(dim=1)[1]  # [B]
        
        # 3. 根据匹配索引获取对应的文本特征
        matched_text = projected_text[matched_idx]  # [B, feat_dim]
        matched_text = matched_text.unsqueeze(1).expand(-1, self.num_points, -1)  # [B, num_points, feat_dim]
        
        # 4. 特征变换
        transformed_contour = self.transform(contour_feat)  # [B, num_points, feat_dim]
        
        # 5. 计算注意力权重
        attn_weights = torch.matmul(contour_feat, matched_text.transpose(-2, -1)) / math.sqrt(self.feat_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, num_points, num_points]
        
        # 6. 特征加权
        attended_text = torch.matmul(attn_weights, matched_text)  # [B, num_points, feat_dim]
        
        # 7. 特征融合
        gate_input = torch.cat([contour_feat, attended_text], dim=-1)
        gate = self.gate(gate_input)
        
        fusion_input = torch.cat([transformed_contour, attended_text], dim=-1)
        fused_feat = self.fusion(fusion_input)
        
        # 8. 残差连接和归一化
        output = contour_feat + gate * fused_feat
        output = self.norm(output)
        
        return output


class Evolution(nn.Module):
    def __init__(self):
        super(Evolution, self).__init__()

        self.fuse = nn.Conv1d(128, 64, 1)
        self.state_compression = VMAMBA2Block(dim=64*2, input_resolution=128)
        self.init_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='vm2')
        self.iter = 2
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='vm2')
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 在初始化时加载模型和分词器
        # try:
        #     model_path = cfg.model_clinical_bert
        #     print(f"Loading ClinicalBERT from: {model_path}")
            
        #     # 检查模型文件是否存在
        #     if not os.path.exists(model_path):
        #         raise ValueError(f"Model path does not exist: {model_path}")
                
        #     # 使用本地缓存
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         model_path,
        #         local_files_only=True,
        #         use_fast=True
        #     )
            
        #     self.ClinicalBERT = AutoModel.from_pretrained(
        #         model_path,
        #         local_files_only=True
        #     )
            
        #     # 验证模型加载
        #     if not hasattr(self.ClinicalBERT, "config"):
        #         raise ValueError("Model failed to load properly")
                
        #     print("ClinicalBERT model loaded successfully")
        #     print(f"Model config: {self.ClinicalBERT.config}")
            
        #     # 冻结参数
        #     for param in self.ClinicalBERT.parameters():
        #         param.requires_grad = False
                
        # except Exception as e:
        #     print(f"Error loading ClinicalBERT: {str(e)}")
        #     raise

        # 替换融合模块
        self.fusion = MultiModalFusion(
            text_dim=768,
            feat_dim=64,
            num_points=128
        )
        self.cross_attention = CrossAttentionFusion(dim=64)
        self.layer_norm = nn.LayerNorm(64)

        # 修改融合模块
        self.text_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )
        
        # 简化的特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True)
        )
        
        # 添加特征压缩模块
        self.feature_compression = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),  # 1x1卷积压缩通道
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # 门控函数
        self.fusion_gate = nn.Sequential(
            nn.Linear(64 * 2, 64),  # 64*2 是因为要考虑文本和视觉特征的拼接
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Sigmoid()  # 输出门控权重
        )

    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch)
        output.update({'i_it_4py': init['i_it_4py'], 'i_it_py': init['i_it_py']})
        output.update({'i_gt_4py': init['i_gt_4py'], 'i_gt_py': init['i_gt_py']})
        return init

    def prepare_training_evolve(self, output, batch, init):
        evolve = snake_gcn_utils.prepare_training_evolve(output['ex_pred'], init)
        output.update({'i_it_py': evolve['i_it_py'], 'c_it_py': evolve['c_it_py'], 'i_gt_py': evolve['i_gt_py']})
        evolve.update({'py_ind': init['py_ind']})
        return evolve

    def prepare_testing_init(self, output):
        init = snake_gcn_utils.prepare_testing_init(output['detection'][..., :4], output['detection'][..., 4])  # init = {'i_it_4py': i_it_4pys  （0，40，2）, 'c_it_4py': c_it_4pys   （0，40，2）, 'ind': ind}
        output['detection'] = output['detection'][output['detection'][..., 4] > snake_config.ct_score]
        output.update({'it_ex': init['i_it_4py']})
        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = snake_gcn_utils.prepare_testing_evolve(ex)
        output.update({'it_py': evolve['i_it_py']})
        return evolve

    def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        #print(i_it_pprepare_testing_initoly.shape)
        if len(i_it_poly) == 0:
            return torch.zeros([0, 4, 2]).to(i_it_poly)  # 这个张量的形状可以被理解为一个包含 0 个四维向量的集合，其中每个四维向量本身又包含 2 个元素。

        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
        ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
        init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
        init_feature = self.fuse(init_feature)

        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        i_poly = i_it_poly + snake(init_input, adj).permute(0, 2, 1)
        i_poly = i_poly[:, ::snake_config.init_poly_num//4]

        return i_poly

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, later_polys=None, cls_ids=None):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
            
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)  # [N, 64, 128]
        
        # 历史信息融合
        features_to_fuse = [init_feature]
        if later_polys is not None:
            for later_poly in later_polys:
                later_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, later_poly, ind, h, w)
                features_to_fuse.append(later_feature)
            fused_feature = torch.cat(features_to_fuse, dim=1)

            # 对最后一个轮廓（当前轮廓）乘以 0.9，其他的历史轮廓乘以 0.1
            num_features = len(features_to_fuse)
            weights = torch.ones(num_features * 64, device=fused_feature.device) * (1 / (len(features_to_fuse)-1))  # 默认历史轮廓权重为 0.1
            weights[-64:] = 1.0 # 当前轮廓（最后一个）权重为 0.9
            # 对每个特征应用权重
            fused_feature = fused_feature * weights.view(1, -1, 1)  # 按照维度加权

            compressed_feature = self.state_compression(fused_feature)
            init_feature = torch.split(compressed_feature, 64, dim=1)[0]

        # 移除文本特征融合部分
        # if cls_ids is not None:
        #     text_features = self.get_text_features(cls_ids, cfg)
            
        #     if text_features is not None:
        #         # print(f"文本特征维度111: {text_features.shape}")
        #         # print(f"轮廓特征维度111: {init_feature.shape}")
                
        #         # 1. 投影文本特征
        #         text_feat = self.text_projection(text_features)  # [N, 64]
                
        #         # 2. 扩展文本特征维度以匹配轮廓特征
        #         text_feat = text_feat.unsqueeze(2).expand(-1, -1, init_feature.shape[2])  # [N, 64, 128]
                
        #         # 3. 转置特征以适应门控函数
        #         init_feat_trans = init_feature.transpose(1, 2)  # [N, 128, 64]
        #         text_feat_trans = text_feat.transpose(1, 2)  # [N, 128, 64]
                
        #         # 4. 计算门控权重
        #         gate_input = torch.cat([init_feat_trans, text_feat_trans], dim=-1)  # [N, 128, 128]
        #         gate = self.fusion_gate(gate_input)  # [N, 128, 64]
                
        #         # 5. 应用门控残差
        #         text_feat_gated = text_feat_trans * gate  # [N, 128, 64]
        #         fused_feature = init_feat_trans + text_feat_gated  # 残差连接
                
        #         # 6. 恢复维度顺序
        #         init_feature = fused_feature.transpose(1, 2)  # [N, 64, 128]

        
        # 确保最终输入维度正确
        c_it_poly = c_it_poly * snake_config.ro  # [N, 128, 2]
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)  # [N, 66, 128]
        
        # 添加维度检查
        # print(f"init_input shape: {init_input.shape}")
        assert init_input.shape[1] == 66, f"输入通道数应为66，但得到{init_input.shape[1]}"
        
        # 获取邻接矩阵
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        
        # Snake网络处理
        offset = snake(init_input, adj)  # 输入维度应为 [N, 66, 128]
        i_poly = i_it_poly * snake_config.ro + offset.permute(0, 2, 1)
        
        return i_poly

    def forward(self, output, cnn_feature, batch=None):
        ret = output
        # batch = {dict : 2}，包括两个部分：batch['inp']=(1,3,544,544),这个应该是原输入图像，batch['meta']={dict:4}，其中{'ann':null, 'center':[256,256], 'scale':[512, 512], 'vis_GT':null}，这个记录的是batch的一些参数
        # output = {dict : 2}, 包括两个部分：ct_hm (1, 9, 136, 136) 和 wh(1, 2, 136, 136)，这两个信息看看怎么用
        # cnn_feature  = {tensor:[1, 64, 136, 136]}

        if batch is not None and 'vis_GT' not in batch['meta']:  # 如果不是测试模式（那就是训练模式呗），则执行下面语句
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            # print("类别ID111:", batch['ct_cls'])
            ex_pred = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['4py_ind'])
            ret.update({'ex_pred': ex_pred, 'i_gt_4py': output['i_gt_4py']})

            # with torch.no_grad():
            #     init = self.prepare_training_evolve(output, batch, init)


            py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['i_it_py'], 
                                       init['c_it_py'], init['py_ind'], 
                                       cls_ids=batch['ct_cls'][0])  # 传入类别信息
            py_preds = [py_pred]

            for i in range(self.iter):
                # py_pred = py_pred / snake_config.ro
                # c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
                # evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                # py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'], py_preds[-1])
                # py_preds.append(py_pred)

                py_pred = py_pred / snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
                evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                history_polys = py_preds[:i+1]  # 第i轮使用第0~i轮的历史
                py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'], later_polys=history_polys)
                py_preds.append(py_pred)

            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro})
            # print(batch['ct_cls'])

        if not self.training:
            with torch.no_grad():
                # 1. 初始化阶段
                init = self.prepare_testing_init(output)
                ex = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['ind'])
                ret.update({'ex': ex})

                # 2. 演化阶段 - 与训练保持一致
                evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))
                
                # 获取当前批次的类别信息
                cls_ids = None
                if batch is not None and 'ct_cls' in batch:
                    cls_ids = batch['ct_cls'][0]
                
                # 第一次演化 - 添加类别信息的支持
                py = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], init['ind'],cls_ids=cls_ids)
                pys = [py]  # 保存未缩放的结果
                
                # 迭代优化 - 与训练一致
                for i in range(self.iter):
                    # 转换为标准化坐标
                    # py_normalized = py / snake_config.ro
                    # c_py = snake_gcn_utils.img_poly_to_can_poly(py_normalized)
                    # evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                    # # 执行演化 - 保持与训练一致
                    # py = self.evolve_poly(evolve_gcn, cnn_feature, py_normalized, c_py, init['ind'], later_poly=pys[-1],  cls_ids=cls_ids)
                    # pys.append(py)

                    py = py / snake_config.ro
                    c_py = snake_gcn_utils.img_poly_to_can_poly(py)
                    evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                    history_polys1 = pys[:i+1]  # 注意推理时保存的是除以ro的
                    py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'], later_polys=history_polys1)
                    pys.append(py)
                
                # 最后统一处理缩放
                pys = [p / snake_config.ro for p in pys]
                ret.update({'py': pys})

        return output
