from lib.networks.darnet.drn_contours import DRNContours
from lib.config import cfg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


# 定义数据集类
class Drn_Dataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # 模型参数存放路径
        self.state_dict = cfg.pretrain_drn.state_dir
        self.image_filenames = [f"{i}_image.jpg" for i in range(cfg.pretrain_drn.image_nums)]
        # 数据路径
        self.train_images_path = []
        self.train_energemaps_path = []
        for i in range(cfg.pretrain_drn.image_nums):
            self.train_images_path.append(cfg.pretrain_drn.train_images_path +'/'+ '{}_image.jpg'.format(i))
            self.train_energemaps_path.append(
                cfg.pretrain_drn.train_images_path +'/'+ '{}_mask.jpg'.format(i))  # 这里用的数据是多mask的标注————>能量图
    def get_filename(self, index):
        return self.image_filenames[index]
    def __len__(self):
        return len(self.train_images_path)

    def __getitem__(self, index):
        # 读取原图和能量图
        img_path = self.train_images_path[index]
        energemap_path = self.train_energemaps_path[index]

        img = cv2.imread(img_path)
        energemap = cv2.imread(energemap_path)

        # 进行变换操作
        if self.transform:
            img = self.transform(img)
            energemap = self.transform(energemap)

        return img, energemap


# 计算SSIM，确保输入是四维张量 (batch_size, channels, height, width)
def ssim_loss(input, target):
    # 将PyTorch tensor转换为numpy数组，确保通道在最后一个维度上
    input_np = input.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # 计算SSIM损失
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_n = ssim(input_np, target_np, win_size=1, channel_axis=0, multichannel=True, use_sample_covariance=False,
                  gaussian_weights=True, sigma=1.5, C1=C1, C2=C2)

    # 由于ssim返回的是单个数值，我们将其扩展到input的形状以进行后续的逐元素相加
    ssim_loss_value = 1 - ssim_n

    return ssim_loss_value


# 我随便定义的损失函数，结合了像素级损失和结构相似性损失
class CustomLoss(nn.Module):
    def __init__(self, pixel_loss_weight=1.0, ssim_loss_weight=1.0):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = ssim_loss
        self.pixel_loss_weight = pixel_loss_weight
        self.ssim_loss_weight = ssim_loss_weight

    def forward(self, pred, target):
        # 计算像素级损失
        pixel_loss = self.mse_loss(pred, target)
        # 计算SSIM损失
        ssim_loss_value = self.ssim_loss(pred, target)
        # 组合损失
        loss = self.pixel_loss_weight * pixel_loss + self.ssim_loss_weight * ssim_loss_value
        return loss


# 实例化模型、损失函数和优化器
model = DRNContours()
criterion = CustomLoss(pixel_loss_weight=1.0, ssim_loss_weight=0.85)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据集和数据加载器
# 定义数据增强和预处理操作   我随便写的，以后可以好好设计一下
transform = transforms.Compose([
    # 将PIL图像或numpy.ndarray转换为Tensor
    transforms.ToTensor(),
    # 归一化操作，根据ImageNet数据集的均值和标准差进行标准化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

train_dataset = Drn_Dataset(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=cfg.pretrain_drn.batch_size, shuffle=True)

# 导入最新模型
model_folder_path = cfg.pretrain_drn.state_dir
model_files = [f for f in os.listdir(model_folder_path) if f.startswith('pretrain_epoch_') and f.endswith('.pth')]  # 获取所有模型文件的名称
if len(model_files) != 0:
    # 提取文件名中的数字并找到最大的数字
    max_epoch = max(int(file.split('_')[-1].split('.')[0]) for file in model_files)
    # 找到包含最大数字的文件
    latest_state = next(file for file in model_files if int(file.split('_')[-1].split('.')[0]) == max_epoch)
    model_path = os.path.join(model_folder_path, latest_state)
    model.load_state_dict(torch.load(model_path))
    print("成功导入上次训练的模型:")
    print(model_path)
    print("------------------------")
else:
    print("无法导入已训练模型参数！")
    print("------------------------")

# 训练循环
num_epochs = 100
save_every = 20  # 每训练20轮保存一次模型参数
had_trained = 0

if len(model_files) != 0:
    had_trained = latest_state.split("_")[-1].split(".")[0]  # 输出为已训练的轮数
    had_trained = int(had_trained)

for epoch in range(num_epochs):
    # 从之前基础上继续训练
    epoch = epoch + had_trained
    if epoch > num_epochs:
        break

    model.train()
    print('第', epoch + 1, '轮······')
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # 检查数据形状，确保每个样本都符合 (3, 512, 512)
        for i in range(data.size(0)):  # 遍历批次中的每个样本
            # 计算文件索引
            filename = train_dataset.get_filename(batch_idx * train_loader.batch_size + i)

            # 检查 data 的形状
            if data[i].shape != torch.Size([3, 512, 512]):
                print(f"Filename: {filename}, Data shape: {data[i].shape}")

        output = model(data)

        for i in range(data.size(0)):  # 再次遍历以确保输出的处理
            filename = train_dataset.get_filename(batch_idx * train_loader.batch_size + i)

            if output[-1].shape != torch.Size([3, 512, 512]):
                print(f"Filename: {filename}, Output shape: {output[-1].shape}")

        output = output[-1].view(3, 512, 512)  # 这里我先只取一层  output:tensor 3,512,512
        target = target[-1].view(3, 512, 512)  # target:tensor 3,512,512
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Train Epoch: {epoch + 1} \t Loss: {loss.item():.6f}")

    # 每训练n轮保存一次模型参数
    if (epoch + 1) % save_every == 0:
        save_path = os.path.join('/home/ub/PycharmProjects/EnergeSnake/data/model/pretrain_drn',
                                 f'pretrain_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model parameters saved at epoch {epoch + 1}.")

print("Model training completed.")

# 运行代码：/data/public/miniconda3/envs/snake3/bin/python pre_train_drn.py