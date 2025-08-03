import cv2
mask = cv2.imread('E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\multiple_segmentdata/new_data/3_mask.png', 0)
print(sum(mask==255))