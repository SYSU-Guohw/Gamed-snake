from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    
    optimizer = make_optimizer(cfg, network)

    scheduler = make_lr_scheduler(cfg, optimizer)

    recorder = make_recorder(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True)

    print("begin_epoch:", begin_epoch, "train.epoch:", cfg.train.epoch)


    if cfg.train_new_data:
        cfg.train.epoch = 1000
        cfg.save_ep = 200
    for epoch in range(begin_epoch, cfg.train.epoch):
        print("第", epoch, "轮···")
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

    return network


def main():
    cfg.train_new_data = True
    network = make_network(cfg)
    train(cfg, network)


if __name__ == "__main__":
    # python train_net.py --cfg_file configs\\sbd_snake.yaml model sbd_snake train_new_data True
    main()
