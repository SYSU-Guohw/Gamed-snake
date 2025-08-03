import os
from PIL import Image

def remove_leading_zeros_from_number(part):
    # 移除数字部分的前导零
    return str(int(part))

def process_filename(filename):
    # 分割文件名和扩展名
    name, ext = os.path.splitext(filename)
    # 分割文件名中的各个部分
    parts = name.split('_')
    # 对每个部分进行处理，如果是数字则去除前导零
    new_parts = [remove_leading_zeros_from_number(part) if part.isdigit() else part for part in parts]
    # 重新组合文件名和扩展名
    new_name = '_'.join(new_parts)
    return new_name + ext
