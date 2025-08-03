from torch.optim.lr_scheduler import MultiStepLR
from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, ManualStepLR

# 不咋用看
# 这段代码提供了创建和设置学习率调度器（LR Scheduler）的函数

def make_lr_scheduler(cfg, optimizer):   # 创建学习率调度器
    if cfg.train.warmup:  # warmup ：初始阶段线性增加学习率，直到预热期结束，然后按照给定的里程碑和衰减因子进行调整
        scheduler = WarmupMultiStepLR(optimizer, cfg.train.milestones, cfg.train.gamma, 1.0/3, 5, 'linear')
    elif cfg.train.scheduler == 'manual':   # manual ：允许用户在训练过程中手动设置学习率的步进
        scheduler = ManualStepLR(optimizer, milestones=cfg.train.milestones, gammas=cfg.train.gammas)
    else:     # 其他情况 ：创建一个标准的MultiStepLR调度器，它在预设的里程碑处按照给定的衰减因子调整学习率
        scheduler = MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma)
    return scheduler


def set_lr_scheduler(cfg, scheduler):   # 设置学习率调度器
    if cfg.train.warmup:
        scheduler.milestones = cfg.train.milestones
    else:
        scheduler.milestones = Counter(cfg.train.milestones)
    scheduler.gamma = cfg.train.gamma

