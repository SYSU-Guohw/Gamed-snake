import cv2
import numpy as np

def create_energy_map(binary_image):
    # 对二值图像应用距离变换，计算每个点到最近非255值点的距离
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    # 将距离变换的结果归一化到0-255范围内
    # 因为我们想要像素值随着距离的增加而减小，所以最大能量值对应距离为0的地方
    max_energy = 255  # 最大能量值
    min_energy = 0    # 最小能量值
    energy_map = cv2.normalize(dist_transform, None, min_energy, max_energy, cv2.NORM_MINMAX).astype(np.uint8)
    energy_map = 255 - energy_map

    return energy_map

# 读取二值图像
binary_image_path = 'E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\multiple_segmentdata\images600_2/0_mask_2.jpg'
binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

# 创建能量图
energy_map = create_energy_map(binary_image)

# 显示结果
cv2.imshow('Energy Map', energy_map)
cv2.waitKey(0)
cv2.destroyAllWindows()