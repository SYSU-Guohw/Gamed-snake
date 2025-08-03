import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel


class Trainer(object):
    def __init__(self, network):   # 这里的 network 指 lib/train/trainers/snake.py 中的 NetworkWrapper 对象（本质上是一个封装好了损失函数的网络模型）
        current_device = torch.cuda.current_device()
        print(f"当前默认的 GPU 设备索引: {current_device}")
        network = network.cuda()   # 将模型迁移到 GPU
        network = DataParallel(network)  # dataParallel 是一个并行处理包装器，它可以在多个 GPU 上并行地复制和训练模型，从而加快训练过程。当使用 DataParallel 时，输入数据会自动分配到各个 GPU 上，然后每个 GPU 上的模型副本独立地进行前向和反向传播，最后将梯度汇总并更新模型参数。

        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v.float()) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        print(len(data_loader))
        self.network.train()  # 将模型设置为训练模式
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            # batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats = self.network(batch)   # 执行 NetworkWrapper 对象的 forward 函数
            # 上句，这里有没有可能定义两个loss : loss1 & loss2 , 分别记录darnet和dla两个输出端的损失
            # 在误差逆传播的过程中分别训练两个头的参数（更新一个头的时候mask掉另一个）————训练方案有待讨论

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)



        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)

