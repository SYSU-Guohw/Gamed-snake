import torch.utils.data as data
import glob
import os
import cv2
import numpy as np
from lib.utils.snake import snake_config
from lib.utils import data_utils
from lib.config import cfg
import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
import sys


image_num = 13
img_path = 'E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\multiple_segmentdata/new_data/'

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.imgs = []
        if os.path.isdir(img_path):
            # import random
            # random.seed(cfg.random_num)
            # random_numbers = random.sample(range(600), 480)
            # full_set = set(range(600))  # 创建全集，即从0到599的所有数字
            # test_random = full_set - set(random_numbers)  # 除去训练集就是测试集
            for i in range(1, image_num+1):
                self.imgs.append(img_path + '{}_image.jpg'.format(i))

            # self.imgs = [cfg.vis_GT.img_path + "/{}_image.jpg".format(i) for i in range(480, 600)]
        elif os.path.exists(img_path):
            self.imgs = [img_path]
        else:
            raise Exception("测试图片文件夹不存在！")

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img = self.imgs[index]
        img = cv2.imread(img)

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1

        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        # inp = { ndarray:(3, 544, 544) }
        inp = self.normalize_image(inp)
        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'vis_GT': '', 'ann': ''}
        ret.update({'meta': meta})

        return ret, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


def poly2mask(ex):
    ex = ex[-1] if isinstance(ex, list) else ex
    ex = ex.detach().cpu().numpy() * 3.77

    img = np.zeros((512, 512))
    ex = np.array(ex)
    ex = ex.astype(np.int32)
    for i in range(ex.shape[0]):
        img = cv2.polylines(img, [ex[i]], True, 1, 1)
        img = cv2.fillPoly(img, [ex[i]], 1)
    return img


def cal_iou(mask, gtmask):
    jiaoji = mask * gtmask
    bingji = ((mask + gtmask) != 0).astype(np.int16)
    return jiaoji.sum() / bingji.sum()


def cal_dice(iou):
    return 2 * iou / (iou + 1)


def TEST():
    visual = True
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    dataset = Dataset()
    # 结果保存路径
    save_root = 'E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\zrc_visual/new_data_result'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    iousum = 0  # iou系数和
    dicesumm = 0  # Dice系数和
    counter = 0  # 用于计数处理了多少个批次的数据

    for batch, img_path in tqdm.tqdm(dataset):
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)  # dict_keys(['ct_hm', 'wh', 'ct', 'detection', 'it_ex', 'ex', 'it_py', 'py'])
            poly = output['py']
            detection = output['detection']
            img = cv2.imread(img_path)
            mask_pre = poly2mask(poly)

            cv2.imshow('Image Window', mask_pre)
            # 等待用户按下任意键，再关闭图片窗口
            cv2.waitKey(0)
            # 关闭所有的OpenCV窗口
            cv2.destroyAllWindows()

            mask_paths = glob.glob(img_path.replace("_image.jpg", "_mask.png"))
            mask_gt = np.zeros((512, 512))
            for maskpath in mask_paths:
                mask = cv2.imread(maskpath, 0)

                # cv2.imshow('Image Window', mask)
                # # 等待用户按下任意键，再关闭图片窗口
                # cv2.waitKey(0)
                # # 关闭所有的OpenCV窗口
                # cv2.destroyAllWindows()

                mask_gt += cv2.imread(maskpath, 0)  # 新数据集中不是所有掩码位置都是255
                # 使用numpy.where()将所有大于0的元素设置为255
                mask_gt = np.where(mask_gt > 0, 255, mask_gt)
                # 再将255————>1    注意！！！一定要将像素转为0—1二值，否则后面计算IOU会有问题
                mask_gt = mask_gt/255
            # cv2.imwrite("E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\zrc_visual\new_data_result\check_mask_gt.jpg",mask)
            # cv2.imwrite("E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\zrc_visual\new_data_result\check_mask_pre.jpg",mask)
            print(cal_iou(mask_pre, mask_gt))
            iousum += cal_iou(mask_pre, mask_gt)
            dicesumm += cal_dice(cal_iou(mask_pre, mask_gt))
            counter += 1

            if visual:
                for j in range(poly[2].shape[0]):
                    poly2visual = 3.75 * np.array(poly[2][j, ...].cpu())  # 在图像 img 上绘制文本。文本内容是类别名称和检测的置信度
                    # 遍历缩放后的多边形 poly2visual 的所有顶点
                    for i in range(poly2visual.shape[0]):  # 在图像上绘制连接多边形顶点的线段
                        cv2.line(img, (int(poly2visual[i, 0]), int(poly2visual[i, 1])), (
                        int(poly2visual[(i + 1) % poly2visual.shape[0], 0]),
                        int(poly2visual[(i + 1) % poly2visual.shape[0], 1])), color=(255, 0, 255), thickness=2)
        if visual:
            cv2.imwrite(save_root + "/{}".format(os.path.basename(img_path)), img)

    print("mIou:{:.4f} mDice:{:.4f}".format(iousum / counter, dicesumm / counter))


TEST()