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


class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.imgs = []
        if os.path.isdir(cfg.test.img_path):
            # # --------------------------------------------
            # # 808测试
            # import random
            # random.seed(cfg.random_num)
            # random_numbers = random.sample(range(808), 646)
            # full_set = set(range(549))  # 创建全集，即从0到599的所有数字
            # test_random = full_set - set(random_numbers)  # 除去训练集就是测试集
            # for i in test_random:
            #     self.imgs.append(cfg.train.data_path +200 '{}_image.jpg'.format(i))
            # # --------------------------------------------

            # self.imgs = [cfg.vis_GT.img_path + "/{}_image.jpg".format(i) for i in range(480, 600)]

            # --------------------------------------------
            # import random
            # random.seed(cfg.random_num)
            # random_numbers = random.sample(range(230), 184)
            # full_set = set(range(230))
            # test_random = full_set - set(random_numbers)  # 除去训练集就是测试集
            # for i in test_random:
            #     self.imgs.append(cfg.train.data_path +'{}_image.jpg'.format(i))

            for i in range(20):
                self.imgs.append(cfg.train.data_path +'{}_image.jpg'.format(i))
            # --------------------------------------------

            # BTCV测试
            # for i in range(1,28):
            #  self.imgs.append(cfg.train.data_path + '{}_image.jpg'.format(i))

        elif os.path.exists(cfg.test.img_path):
            self.imgs = [cfg.test.img_path]
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
        inp = img

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1

        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        inp = self.normalize_image(inp)  # inp = { ndarray:(3,544, 544) }

        # 将(3, 544, 544)的图片截取为（3,512,512）
        # =====================================
        # 计算中心裁剪的起始点
        # start_x = (544 - 512) // 2
        # start_y = (544 - 512) // 2
        #
        # # 计算裁剪区域的结束点
        # end_x = start_x + 512
        # end_y = start_y + 512
        #
        # # 使用切片操作裁剪图片
        # inp = inp[:, start_x:end_x, start_y:end_y]
        # # =====================================

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
    visual = 1
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    dataset = Dataset()
    # 结果保存路径
    #save_root = '/home/ub/PycharmProjects/EnergeSnake/zrc_visual/BTCV'
    save_root = '/data/lyc/EnergeSnake/zrc_visual/230_5_5'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    #class_list = ["S", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]
    class_list = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
                  ,"", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
                  ,"", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
    iousum = 0  # iou系数和
    dicesumm = 0  # Dice系数和
    counter = 0  # 用于计数处理了多少个批次的数据

    class_colors = {
    0: (255, 0, 0),      # 红色
    1: (0, 255, 0),      # 绿色
    2: (0, 0, 255),      # 蓝色
    3: (255, 255, 0),    # 黄色
    4: (255, 0, 255),    # 洋红色
    5: (0, 255, 255),    # 青色
    6: (128, 0, 0),      # 暗红色
    7: (0, 128, 0),      # 暗绿色
    8: (0, 0, 128),      # 深蓝色
    9: (128, 128, 0),    # 橄榄色
    10: (128, 0, 128),   # 紫色
    11: (0, 128, 128),   # 深青色
    12: (192, 192, 192), # 银色
    13: (128, 128, 128), # 灰色
    14: (255, 165, 0),   # 橙色
    15: (255, 20, 147),  # 深粉色
    16: (75, 0, 130),    # 靛蓝色
    17: (173, 216, 230), # 浅蓝色
    18: (34, 139, 34),   # 森林绿
    19: (255, 215, 0),   # 金色
    20: (219, 112, 147), # 浅粉色
    21: (255, 99, 71),   # 番茄红
    22: (154, 205, 50),  # 黄绿色
    23: (85, 107, 47),   # 深橄榄绿
    24: (139, 69, 19),   # 马鞍棕色
    25: (189, 183, 107), # 深卡其色
    26: (240, 230, 140), # 卡其色
    27: (250, 250, 210), # 浅黄色
    28: (230, 230, 250), # 薰衣草色
    29: (216, 191, 216), # 蓟色
    30: (255, 228, 225), # 薄雾玫瑰
    31: (240, 128, 128), # 浅珊瑚色
    32: (255, 160, 122), # 浅鲑鱼色
    33: (255, 127, 80),  # 珊瑚色
    34: (255, 69, 0),    # 橙红色
    35: (255, 140, 0),   # 深橙色
    36: (184, 134, 11),  # 深金黄色
    37: (218, 165, 32),  # 金菊色
    38: (238, 232, 170), # 浅金菊色
    39: (189, 183, 107), # 深卡其布色
    40: (143, 188, 143), # 深海洋绿
    41: (102, 205, 170), # 中海洋绿
    42: (32, 178, 170),  # 浅海洋绿
    43: (0, 139, 139),   # 深青色
    44: (100, 149, 237), # 矢车菊蓝
    45: (25, 25, 112),   # 午夜蓝
    46: (72, 61, 139),   # 深板岩蓝
    47: (106, 90, 205),  # 板岩蓝
    48: (123, 104, 238), # 中板岩蓝
    49: (147, 112, 219), # 中紫色
    50: (139, 0, 139),   # 深洋红色
    51: (148, 0, 211),   # 深紫色
    52: (153, 50, 204),  # 暗兰花紫
    53: (186, 85, 211),  # 中兰花紫
    54: (128, 0, 0),     # 栗色
    55: (165, 42, 42),   # 褐色
    56: (178, 34, 34),   # 火砖色
    57: (205, 92, 92),   # 印度红
    58: (220, 20, 60),   # 猩红色
    59: (255, 0, 0),      # 红色
    60: (18, 34, 34),   
    61: (205, 9, 9), 
    62: (20, 200, 60),  
    63: (0, 30, 200)     
    }

    for batch, img_path in tqdm.tqdm(dataset):
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)  # dict_keys(['ct_hm', 'wh', 'ct', 'detection', 'it_ex', 'ex', 'it_py', 'py'])
            poly = output['py']
            # poly = [4*p for p in poly]
            detection = output['detection']
            img = cv2.imread(img_path)
            mask_pre = poly2mask(poly)
            mask_paths = glob.glob(img_path.replace("_image.jpg", "_mask") + "*")
            mask_gt = np.zeros((512, 512))
            for maskpath in mask_paths:
                # print(maskpath)
                mask = cv2.imread(maskpath, 0)
                mask_gt += cv2.imread(maskpath, 0) / 255
            # cv2.imwrite("E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\zrc_visual\new_data_result\check_mask_gt.jpg",mask_gt*255)
            # cv2.imwrite("E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\zrc_visual\new_data_result\check_mask_pre.jpg",mask_pre*255)
            print(cal_iou(mask_pre, mask_gt))
            iousum += cal_iou(mask_pre, mask_gt)
            dicesumm += cal_dice(cal_iou(mask_pre, mask_gt))
            counter += 1

            if visual:
                for j in range(poly[2].shape[0]):

                    poly2visual = 3.75 * np.array(poly[2][j, ...].cpu())
                    class_id = int(detection[j, 5])  # 类别 ID
                    class_name = class_list[class_id]  # 类别名称
                    # print(class_id)
                    confidence = detection[j, 4]  # 置信度
                    # 在图像上绘制类别名称和置信度
                    # cv2.putText(img, f"{class_id}_{confidence:.2f}",
                    #             (int(3.75 * (detection[j, 0] + detection[j, 2]) / 2),
                    #              int(3.75 * (detection[j, 1] + detection[j, 3]) / 2)),
                    #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)

                    # poly2visual = 3.75 * np.array(poly[2][j, ...].cpu())  # 在图像 img 上绘制文本。文本内容是类别名称和检测的置信度
                    # cv2.putText(img, class_list[int(detection[j, 5])] + "_" + "{:.2f}".format(detection[j, 4]), (
                    # int(3.75 * (detection[j, 0] + detection[j, 2]) / 2),
                    # int(3.75 * (detection[j, 1] + detection[j, 3]) / 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    #             (100, 200, 200), 1)
                    # 遍历缩放后的多边形 poly2visual 的所有顶点

                    # 定义每个类别的颜色（可以根据类别数量扩展）

                    for i in range(poly2visual.shape[0]):  # 在图像上绘制连接多边形顶点的线段

                        color1 = class_colors[class_id]
                        cv2.line(img, (int(poly2visual[i, 0]), int(poly2visual[i, 1])), (
                        int(poly2visual[(i + 1) % poly2visual.shape[0], 0]),
                        int(poly2visual[(i + 1) % poly2visual.shape[0], 1])), color=color1, thickness=2)
        if visual:
            cv2.imwrite(save_root + "/{}".format(os.path.basename(img_path)), img)

    print("mIou:{:.4f} mDice:{:.4f}".format(iousum / counter, dicesumm / counter))