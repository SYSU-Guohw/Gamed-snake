from .trainer import Trainer
import imp
import os


def _wrapper_factory(cfg, network):
    # module 和 path 分别是模块的名称和该模块文件的路径
    module = '.'.join(['lib.train.trainers', cfg.task])
    path = os.path.join('lib/train/trainers', cfg.task+'.py')   # lib/train/trainers/snake.py
    # # 点号（.）表示访问模块内部的某个属性或方法。在这它用于访问之前通过 load_source 加载的模块中的 NetworkWrapper 类，从而创建了一个 NetworkWrapper 类的实例
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network)
    return network_wrapper


def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network)
    return Trainer(network)

'''
这个项目简直就是套娃······
  make_trainer的作用大概如下：
      从（）传入 make_trainer 函数一个network，首先调用 _wrapper_factory 函数，_wrapper_factory会找到相应任务下的训练器封装文件（暂且叫这个）（比如lib/train/trainers/snake.py），
    并通过imp（）方法提取训练器封装文件中的NetworkWrapper对象，将network作为NetworkWrapper对象的初始化参数，从而返回一个NetworkWrapper实例，这个实例与输入前的network相比，增加了损失函数的计算和记录的方法。
      然后将第一次封装后的对象作为初始化参数，建立一个Trainer类（实际在进行二次封装），Trainer类中又添加正向传播、误差逆传播、评估等功能。
'''