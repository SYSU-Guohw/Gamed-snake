import cv2
import numpy as np
from tqdm import tqdm

def create_energy_map(binary_image, threshold):
    # 二值化图像，确保0代表轮廓，255代表背景
    binary_image = 255 - cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    '''
    cv2.imshow('binary_image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # 计算距离变换，得到每个点到最近轮廓边界的距离
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    # 截断距离————灵感来自VGN抓取的三维点云
    dist_transform = np.where(dist_transform > threshold, threshold, dist_transform)

    # 初始化能量图，其大小与输入图像相同，并且所有值都设置为0
    energy_map = np.zeros_like(binary_image, dtype=np.uint8)

    # 归一化距离变换的结果到0-255范围，但要确保超过阈值的距离为0
    # 这里我们从255开始减少，因为我们要计算的是到轮廓边界的距离
    normalized_distances = 255 - cv2.normalize(dist_transform, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # 应用阈值，超过阈值的距离设置为0
    energy_map = np.where(dist_transform <= threshold, normalized_distances, 0)
    #energy_map = 255 - energy_map

    return energy_map

for i in tqdm(range(600)):
    # 读取图片
    #binary_image_path = 'E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\multiple_segmentdata\images600_2/0_mask_2.jpg'
    binary_image_path = 'E:/PyCharm-ZRC/PyCharm-Project/DeepSnake_remove_c/multiple_segmentdata/images600/' + str(i) + '_mask.jpg'
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    # 二值化处理，将大于10的像素点变为白色，其余为黑色
    _, binary_image = cv2.threshold(binary_image, 10, 255, cv2.THRESH_BINARY)

    # 创建一个与原图同样大小的空白掩码图
    mask = np.zeros_like(binary_image, dtype=np.uint8)

    # 使用findContours函数寻找轮廓
    # 根据OpenCV的版本，findContours可能返回2个或3个值，我的版本返回3个值，需要提取后两个，否则报错
    contours, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # 检查是否找到轮廓
    if contours:
    # 绘制外层轮廓
        cv2.drawContours(mask, contours, -1, (255), 1)
    else:
        print("没有找到轮廓")

    # 设置距离阈值
    threshold = 50

    # 创建能量图
    energy_map = create_energy_map(mask, threshold)

    cv2.imwrite('E:/PyCharm-ZRC/PyCharm-Project/DeepSnake_remove_c/multiple_segmentdata/images600/' + str(i) + '_energe_map_3.jpg', energy_map)

# # 显示结果
# cv2.imshow('Energy Map', energy_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()