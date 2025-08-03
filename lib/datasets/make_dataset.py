from .transforms import make_transforms
from . import samplers
from .dataset_catalog import DatasetCatalog
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator


torch.multiprocessing.set_sharing_strategy('file_system')


def _dataset_factory(data_source, task):
    module = '.'.join(['lib.datasets', data_source, task])
    path = os.path.join('lib/datasets', data_source, task+'.py')
    dataset = imp.load_source(module, path).Dataset   # lib/datasets/sbd/snake.py
    return dataset
    # 动态加载类， 函数返回 Dataset 类，而不是类的实例，用的时候要创建实例化对象。

def make_dataset(cfg, dataset_name, transforms, is_train=True):
    args = DatasetCatalog.get(dataset_name)
    ''' :args 长这个样子：    'SbdTrain': {
                              'id': 'sbd',
                              'data_root': 'data/sbd/img',
                              'ann_file': 'data/sbd/annotations/sbd_train_instance.json',
                                'split': 'train'
                                  },'''
    data_source = args['id']
    dataset = _dataset_factory(data_source, cfg.task)
    del args['id']  # 删除args字典中的元素（id为键）
    # args['cfg'] = cfg
    # args['transforms'] = transforms
    # args['is_train'] = is_train
    dataset = dataset(**args)  # 创建 Dataset 类的实例化对象
    return dataset


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset

    transforms = make_transforms(cfg, is_train)
    dataset = make_dataset(cfg, dataset_name, transforms, is_train)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter)
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )

    return data_loader
