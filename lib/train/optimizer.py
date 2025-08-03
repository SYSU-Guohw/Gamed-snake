import torch
from lib.utils.optimizer.radam import RAdam

# 该文件定义了训练过程的优化器，没啥东西
# 唯一值得注意的是 weight_decay ：权重衰减（weight decay）参数，这个配置可以写在我们的文章里

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net):
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay  # 权重衰减（weight decay）参数，就是正则化项前的系数

    for key, value in net.named_parameters():
        if not value.requires_grad:  # 如果参数不需要梯度，则使用continue语句跳过当前循环，不对其进行优化器配置
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
