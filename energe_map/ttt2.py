import cv2
import numpy as np
from tqdm import tqdm

def create_external_energy_map(binary_image, threshold):
    # 二值化图像，确保0代表前景（轮廓），255代表背景
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # 对二值图像取反，使得前景（轮廓）为0，背景为255
    binary_image = cv2.bitwise_not(binary_image)

    # 计算距离变换，得到背景到前景的最短距离
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    # 截断距离————灵感来自VGN抓取的三维点云
    dist_transform = np.where(dist_transform > threshold, threshold, dist_transform)

    # 归一化距离变换的结果，得到0-255之间的能量值
    normalized_distances = cv2.normalize(dist_transform, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # 对于超过阈值的距离，将能量值设置为最小值0
    energy_map = np.where(dist_transform > threshold, 0, normalized_distances)

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
    energy_map = create_external_energy_map(binary_image, threshold)
    # 保存能量图
    cv2.imwrite('E:/PyCharm-ZRC/PyCharm-Project/DeepSnake_remove_c/multiple_segmentdata/images600/' + str(i) + '_energe_map_2.jpg', energy_map)

# 显示结果
# cv2.imshow('External Energy Map', energy_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()