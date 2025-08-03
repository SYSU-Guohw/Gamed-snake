import os
import imp


_network_factory = {
}


def get_network(cfg):
    arch = cfg.network   # network: 'ro_34'
    heads = cfg.heads    # heads: {'ct_hm': 9, 'wh': 2}
    head_conv = cfg.head_conv
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _network_factory[arch]
    network = get_model(num_layers, heads, head_conv)
    return network


def make_network(cfg):
    module = '.'.join(['lib.networks', cfg.task])  # task: 'snake'
    path = os.path.join('lib/networks', cfg.task, '__init__.py')
    print("网络路径：", path)
    print("模型ID:", cfg.task)
    print("model:", cfg.model)
    return imp.load_source(module, path).get_network(cfg)
    # 注意！！！这里的 get_network() 函数不是上面定义的的那个，而是 imp.load_source(module, path) 引入进来的模块的类函数
    # 此处的 get_network() 函数来自lib\\networks\\snake 文件
