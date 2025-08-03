import torch.nn as nn
from lib.utils import net_utils
import torch
from lib.config import cfg
import sys
import torch.nn.functional as F

def intermediate_signal(gtpoly):
    decerate = [0,0.25]
    circle_rate=1
    sup_polys=[]
    for rate in decerate:
        for i in range(gtpoly.shape[0]):
            gtpyiter = gtpoly[i,:,:]      #128*2
            # 首先计算poly的中心 扩展成128
            center = torch.mean(gtpyiter,dim=0)
            center_vector = center.repeat(128,1)
            #计算每个点到中心的欧式距离 以及均值
            pdist = nn.PairwiseDistance(p=2)
            pdist_result = pdist(gtpyiter,center_vector)
            mean_pdist = torch.mean(pdist_result)
            #计算poly中心点向每个点发从的单位矢量 并乘以均值得到一个圆
            vector_poly = gtpyiter-center
            pdist_result1 = pdist_result.unsqueeze(dim=1).repeat(1,2)         
            vector_poly_circle = mean_pdist*vector_poly/pdist_result1
            #计算每个点到center距离与均值的差值
            gap_dist = pdist_result-mean_pdist
            #计算缩放权重
            percentage1 = 1+ rate*gap_dist/mean_pdist
            percentage1 = percentage1.unsqueeze(dim=1)
            percentage1 = percentage1.repeat(1,2)
            percentage1 = circle_rate*percentage1
            #根据缩放权重计算简化后的值
            if i==0:
                layer1_sup_poly = vector_poly_circle.mul(percentage1)+center_vector
                layer1_sup_poly = layer1_sup_poly.unsqueeze(dim=0)
            else:
                layer1_sup_poly_toappend = vector_poly_circle.mul(percentage1)+center_vector
                layer1_sup_poly_toappend = layer1_sup_poly_toappend.unsqueeze(dim=0)
                layer1_sup_poly = torch.cat((layer1_sup_poly,layer1_sup_poly_toappend),dim=0)
        sup_polys.append(layer1_sup_poly)
    return sup_polys

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.ct_crit = net_utils.FocalLoss()  # 由 Tsung-Yi Lin 等人在 2017 年的论文《Focal Loss for Dense Object Detection》中提出。它的目的是解决类别不平衡问题，特别是在背景类别（负样本）远多于前景类别（正样本）的情况下
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.reg_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        loss = 0

        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])   # 损失计算
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])   # 损失计算
        scalar_stats.update({'wh_loss': wh_loss})
        loss += 0.1 * wh_loss

        ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
        scalar_stats.update({'ex_loss': ex_loss})
        loss += ex_loss

        # # 预测类别概率（logits）
        # seg_classes = output["seg_classes"]  # 这个是概率预测向量
        # # det_classes = output["detect_classes"].long()  # 这个是类别数字[15,67,···]
        #
        # ct_cls = batch['ct_cls']
        # gt_num = batch['meta']['ct_num']
        # # gt_01 = batch['ct_01'].byte()
        # gt_classes = torch.cat([ ct_cls[i,:gt_num[i]] for i in range(gt_num.size(0))], dim=0)
        #
        # # gt_classes 转换为 One-Hot 编码
        # gt_classes = F.one_hot(gt_classes,
        #                       num_classes=cfg.heads.ct_hm)  # [batch, num_preds, num_classes]
        # consistency_loss = F.kl_div(
        #     F.log_softmax(seg_classes, dim=-1),  # 预测概率（log）
        #     gt_classes,  #  One-Hot
        #     reduction='batchmean'  # 对整个 batch 平均
        # ).long()
        #
        # scalar_stats.update({'con_loss': consistency_loss})
        # loss += 0.01*consistency_loss










































        py_loss = 0
        #output['py_pred'] = [output['py_pred'][-1]]
        if cfg.multistage:
            sup_polys=intermediate_signal(output['i_gt_py'])
            supsignal = [sup_polys[0],sup_polys[1],output['i_gt_py']]
            #print(len(output['py_pred'][0]))
            #print(len(output['py_pred'][0]),len(supsignal[0]))
            for i in range(len(output['py_pred'])):
                py_loss += self.py_crit(output['py_pred'][i], supsignal[2]) / len(output['py_pred'])
        else:
            output['py_pred'] = [output['py_pred'][-1]]
            for i in range(len(output['py_pred'])):
                py_loss += self.py_crit(output['py_pred'][i], output['i_gt_py']) / len(output['py_pred'])
        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

