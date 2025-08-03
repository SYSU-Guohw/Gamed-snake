#  -*- coding: utf-8 -*- 

import cv2
import os
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt 

def uniformsample(pgtnp_px2, newpnum):
    #print(pgtnp_px2)
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)
    #print(edgeidxsort_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2

    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

# 这个函数的作用是保证轮廓闭合（轮廓的第一个点和最后一个点能对上）
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


# 这个函数用于将二值掩码转化为多边形，很重要
def binary_mask_to_polygon(root, tolerance=0):
    # new_data_result 中的二值不是0和1，而是0和255
    mask = cv2.imread(root,0)
    mask = np.array(mask)
    mask = mask/128  # 255/128 = 1
    binary_mask = mask.astype(np.int)
    polygons = []
    # pad new_data_result to close contours of shapes which start and end at an edge  填充掩码以闭合那些在边缘开始和结束的形状的轮廓
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)   # 使用 measure.find_contours 函数寻找填充后的二进制掩码中的所有轮廓。

    import matplotlib.pyplot as plt

    # map = np.zeros([1550, 1550])
    # for i in range(len(contours)):
    #
    #     for point_num in range(len(contours[i])):
    #         point = np.round(contours[i][point_num]).astype(np.int)
    #
    #         map[point[0],point[1]] += 1
    #
    # # 将ndarray转换为Pillow图像
    # from PIL import Image
    # image = Image.fromarray(map, 'L')  # 'L' 模式表示灰度图

    # # 保存图像
    # image.save(root+'counter111.png')

    # # 因为matplotlib不能直接显示形状为(512, 512, 1)的图像，我们需要去掉最后一个维度
    # image = padded_binary_mask.squeeze()
    #
    # # 使用matplotlib显示图像
    # plt.imshow(image, cmap='gray')  # 使用灰度色图显示
    # plt.colorbar()  # 显示色标
    # plt.show()

    #唐梓轩添加，因为默认处理单连通区域，所以这里取最大的单连通区域
    #max=0
    #if len(contours)!=1:
    #    for i,counter1 in enumerate(contours):
    #        if counter1.shape[0]>max:
    #            max=counter1.shape[0]
    #            contours=[counter1]
        
    contours = np.subtract(contours, 1)  # 将轮廓中的每个点的坐标减去1，以补偿之前添加的填充。
    for contour in contours:
        contour = close_contour(contour)  # 这一步的作用是保证轮廓闭合（轮廓的第一个点和最后一个点能对上）
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    polys=[]
    # 下面的 for 循环是为了整理轮廓的坐标格式为（x,y）型，不用细看
    for j in range(len(polygons)):
        poly=[]
        for i in range(int(len(polygons[j])/2)):
            poly.append([polygons[j][2*i],polygons[j][2*i+1]])
        polys.append([np.array(poly)])
    
    return polys




if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    #root = 'E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\multiple_segmentdata\images600/0_mask.jpg'	# 修改为你对应的文件路径
    root = 'E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\multiple_segmentdata\images600_2/1_mask_2.jpg'
    poly = binary_mask_to_polygon(root)
    #print(len(poly))
    #print(len(poly[0][0]))  # 输出331，这个掩膜轮廓上有331个点
    #img = cv2.imread(root)
    #for num in range(len(poly)):
    #    poly2visual=poly[num]
    #    print(poly2visual,poly2visual.shape)
    #    for i in range(poly2visual.shape[0]):
    #        cv2.line(img, (int(poly2visual[i,0]),int(poly2visual[i,1])), (int(poly2visual[(i+1)%poly2visual.shape[0],0]),int(poly2visual[(i+1)%poly2visual.shape[0],1])), color=(255, 0, 255), thickness=2)
    #cv2.imwrite('/data/tzx/snake-master/visual_result/poly.jpg',img)
