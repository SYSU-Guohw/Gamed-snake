from lib.utils.snake import snake_config
from .ct_snake import get_network as get_ro  # get_ro为一个函数对象


_network_factory = {
    'ro': get_ro
}

# E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\lib\networks\make_network.py 中调用的 get_network函数来自这里
def get_network(cfg):
    arch = cfg.network    # cfg.network = 'dla_34'
    heads = cfg.heads     # cfg.heads: {'ct_hm': 9, 'wh': 2}
    head_conv = cfg.head_conv
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _network_factory[arch]  # get_model 即为 get_ro 函数
    network = get_model(num_layers, heads, head_conv, snake_config.down_ratio, cfg.det_dir)  # 向 get_model 函数（即get_ro函数）中传入参数，返回一个拼接后的网络对象
    return network

