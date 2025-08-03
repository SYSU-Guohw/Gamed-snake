import cv2
import numpy as np


def create_rgb_energy_map(binary_image, threshold):
    # 计算距离变换，得到每个点到最近轮廓边界的距离
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    # 截断距离，超过阈值的距离设置为阈值
    dist_transform = np.where(dist_transform > threshold, threshold, dist_transform)

    # 归一化距离变换的结果到0-255范围
    normalized_distances = cv2.normalize(dist_transform, None, alpha=0, beta=255,
                                               norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    energy_map = np.where(dist_transform <= threshold, normalized_distances, 0)

    # # 创建一个与输入图像同样大小的RGB图像
    # rgb_energy_map = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    #
    # # 使用颜色映射转换归一化的距离到RGB颜色
    # # 红色通道表示距离，绿色通道和蓝色通道渐变以提供对比度
    # rgb_energy_map[..., 0] = normalized_distances  # 红色通道
    # rgb_energy_map[..., 1] = normalized_distances  # 绿色通道
    # rgb_energy_map[..., 2] = normalized_distances  # 蓝色通道

    # RGB在opencv中存储为BGR的顺序,数据结构为一个3D的numpy.array,索引的顺序是行,列,通道:
    B = normalized_distances*255
    G = normalized_distances*255
    R = normalized_distances*255
    # 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。
    g = energy_map[:]
    p = 0.2989;
    q = 0.5870;
    t = 0.1140
    B_new = (g - p * R - q * G) / t
    B_new = np.uint8(B_new)
    rgb_energy_map = np.zeros((energy_map.shape[0], energy_map.shape[1], 3), dtype=np.uint8)
    rgb_energy_map[:, :, 0] = B_new
    rgb_energy_map[:, :, 1] = G
    rgb_energy_map[:, :, 2] = R


    return rgb_energy_map


# 读取图片
binary_image_path = 'E:/PyCharm-ZRC/PyCharm-Project/DeepSnake_remove_c/multiple_segmentdata/images600_2/0_mask_2.jpg'
binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

if binary_image is not None:
    # 二值化处理，确保0代表轮廓，255代表背景
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # 设置距离阈值
    threshold = 50  # 根据实际情况调整

    # 创建RGB能量图
    rgb_energy_map = create_rgb_energy_map(binary_image, threshold)

    # 显示结果
    cv2.imshow('RGB Energy Map', rgb_energy_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: 图片没有正确加载。请检查路径是否正确。")