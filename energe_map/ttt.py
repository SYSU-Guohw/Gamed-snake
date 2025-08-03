import cv2
import time
import numpy as np

# 读取图片
omask = cv2.imread('/multiple_segmentdata/images600_2/0_mask_2.jpg')

# cv2.imshow('Image', new_data_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# time.sleep(10)

# mask中只有0和255两种值
print(omask.shape)  # 512,512,3  不是灰度图？

R = omask[:,:, 0]
G = omask[:,:, 1]
B = omask[:,:, 2]
print(np.count_nonzero(omask))
print(np.count_nonzero(R))
print(np.count_nonzero(G))
print(np.count_nonzero(B))
are_equal = np.array_equal(R, G)
print(are_equal)  # True 说明mask的RGB三通道是一样的，可以等效为灰度图



# 二值化处理，将255的像素点变为白色，其余为黑色
_, binary_image = cv2.threshold(R, 127, 255, cv2.THRESH_BINARY)

# 创建一个与原图同样大小的空白掩码图
mask = np.zeros_like(binary_image, dtype=np.uint8)

# 使用findContours函数寻找轮廓
# 根据OpenCV的版本，findContours可能返回2个或3个值
contours, hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

# 检查是否找到轮廓
if contours:
    # 绘制外层轮廓
    cv2.drawContours(mask, contours, -1, (255), 1)
else:
    print("没有找到轮廓")


# 显示原图、二值化图像和掩码图
cv2.imshow('Original Image', R)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Contour Mask', mask)

# 等待用户按下任意键，再关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


