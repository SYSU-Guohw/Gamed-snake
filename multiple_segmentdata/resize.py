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

def convert_png_to_jpg(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                # 将图片转换为jpg格式并保存
                img_converted = img.convert('RGB')
                new_filename = process_filename(filename)
                jpg_filename = new_filename.replace('.png', '.jpg')
                img_converted.save(os.path.join(folder_path, jpg_filename))
                print(f"Converted and saved: {jpg_filename}")
            # 删除原始的.png文件
            os.remove(img_path)
            print(f"Deleted original: {filename}")

def resize_images_in_folder(folder_path, size=(512, 512)):
    # 确保输出文件夹存在
    output_folder = os.path.join(folder_path, 'resized')
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                # 调整图片大小
                img_resized = img.resize(size)
                # 保存调整后的图片
                new_filename = process_filename(filename)
                resized_filename = new_filename
                img_resized.save(os.path.join(output_folder, resized_filename))
                print(f"Resized and saved: {resized_filename}")

if __name__ == "__main__":
    folder_path = input("请输入图片文件夹路径: ")
    convert_png_to_jpg(folder_path)
    resize_images_in_folder(folder_path)