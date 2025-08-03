import os

def rename_jpg_files(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否以.jpg结尾
        if filename.endswith(".jpg"):
            print(filename.split('_')[-1])
            if filename.split('_')[-1] == 'mask.jpg':
                # 构造新的文件名
                new_filename = filename[:-4] + '_1.jpg'  # 去掉原文件名的.jpg后缀，加上_1.jpg
                # 构造完整的文件路径
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_filename)
                # 重命名文件
                os.rename(old_file, new_file)
                print(f'Renamed "{filename}" to "{new_filename}"')

# 使用示例
# 替换下面的路径为你的目标文件夹路径
directory_path = '/home/ub/PycharmProjects/EnergeSnake/multiple_segmentdata/lungmask/resized/'
rename_jpg_files(directory_path)
