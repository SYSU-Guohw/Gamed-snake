import cv2
import numpy as np
from tqdm import tqdm

def create_energy_map(binary_image, threshold):
    # 二值化图像，确保255代表前景，0代表背景
    binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)[1]

    # 计算距离变换，得到背景到前景的最短距离
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    # 初始化能量图
    energy_map = np.zeros_like(binary_image, dtype=np.uint8)

    # 计算轮廓内部的能量值
    internal_energy = cv2.normalize(dist_transform, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    internal_energy[dist_transform <= threshold] = 255  # 轮廓内部且在阈值内的能量值为255

    # 计算轮廓外部的能量值
    external_energy = cv2.normalize(dist_transform, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    external_energy[dist_transform > threshold] = 0  # 轮廓外部且超过阈值的能量值为0

    # 结合内部和外部的能量图
    energy_map = np.where(binary_image == 0, internal_energy, external_energy)

    energy_map = 255 - energy_map
    return energy_map

# 循环
for i in tqdm(range(600)):
    # 读取图片
    #binary_image_path = 'E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\multiple_segmentdata\images600_2/0_mask_2.jpg'
    binary_image_path = 'E:/PyCharm-ZRC/PyCharm-Project/DeepSnake_remove_c/multiple_segmentdata/images600/' + str(i) + '_mask.jpg'
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    # 二值化处理，将大于10的像素点变为白色，其余为黑色
    _, binary_image = cv2.threshold(binary_image, 10, 255, cv2.THRESH_BINARY)

    # 设置距离阈值
    threshold = 50
    # 创建能量图
    energy_map = create_energy_map(binary_image, threshold)
    # 保存能量图
    cv2.imwrite('E:/PyCharm-ZRC/PyCharm-Project/DeepSnake_remove_c/multiple_segmentdata/images600/' + str(i) + '_energe_map_1.jpg', energy_map)

    # # 显示结果
    # cv2.imshow('Energy Map', energy_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

