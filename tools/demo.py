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

        if os.path.isdir(cfg.demo_path):
            self.imgs = [cfg.demo_path+"{}_image.jpg".format(i) for i in range(0,808)]
        elif os.path.exists(cfg.demo_path):
            self.imgs = [cfg.demo_path]
        else:
            raise Exception('NO SUCH FILE')

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

    img = np.zeros((512,512))
    ex = np.array(ex)
    ex = ex.astype(np.int32)
    for i in range(ex.shape[0]):
        img = cv2.polylines(img,[ex[i]],True,1,1)
        img = cv2.fillPoly(img, [ex[i]], 1)
    return img


def cal_iou(mask, gtmask):
    jiaoji = mask*gtmask
    bingji = ((mask+gtmask)!=0).astype(np.int16)
    return jiaoji.sum()/bingji.sum()

def cal_dice(iou):
    return 2*iou/(iou+1)

def demo():
    visual = True
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()
    #print()
    dataset = Dataset()
    # 结果保存路径
    save_root=cfg.demo_vis + "/{}".format(os.path.basename(cfg.model_dir))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    class_list=["S","L5","L4","L3","L2","L1","T12","T11","T10"]
    iousum=0  # iou系数和
    dicesumm=0  # Dice系数和
    counter=0  # 用于计数处理了多少个批次的数据

    for batch,img_path in tqdm.tqdm(dataset):
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch) # dict_keys(['ct_hm', 'wh', 'ct', 'detection', 'it_ex', 'ex', 'it_py', 'py'])
            poly = output['py'] 
            detection = output['detection']
            img = cv2.imread(img_path)
            mask_pre = poly2mask(poly)
            mask_paths=glob.glob(img_path.replace("_image.jpg","_mask")+"*")
            mask_gt = np.zeros((512,512))
            for maskpath in mask_paths:
                mask=cv2.imread(maskpath,0)
                mask_gt += cv2.imread(maskpath,0)/255
            #cv2.imwrite("E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\zrc_visual\new_data_result\check_mask_gt.jpg",mask_gt*255)
            #cv2.imwrite("E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\zrc_visual\new_data_result\check_mask_pre.jpg",mask_pre*255)
            print(cal_iou(mask_pre,mask_gt))
            iousum+=cal_iou(mask_pre,mask_gt)
            dicesumm+=cal_dice(cal_iou(mask_pre,mask_gt))
            counter+=1
            
            '''
            #保存点到npz文件当中
            if visual:
                poly2save = poly[2].cpu()
                poly2save = poly2save.numpy()
                np.savez(save_root+"/{}".format(os.path.basename(img_path).replace("jpg","npz")), 3.75*poly2save)
            '''

            if visual:
                for j in range(poly[2].shape[0]):
                    poly2visual=3.75*np.array(poly[2][j,...].cpu())          # 在图像 img 上绘制文本。文本内容是类别名称和检测的置信度
                    cv2.putText(img, class_list[int(detection[j,5])]+"_"+"{:.2f}".format(detection[j,4]), (int(3.75*(detection[j,0]+detection[j,2])/2), int(3.75*(detection[j,1]+detection[j,3])/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
                    # 遍历缩放后的多边形 poly2visual 的所有顶点
                    for i in range(poly2visual.shape[0]):                    # 在图像上绘制连接多边形顶点的线段
                        cv2.line(img, (int(poly2visual[i,0]),int(poly2visual[i,1])), (int(poly2visual[(i+1)%poly2visual.shape[0],0]),int(poly2visual[(i+1)%poly2visual.shape[0],1])), color=(255, 0, 255), thickness=2)
        if visual:
            cv2.imwrite(save_root+"/{}".format(os.path.basename(img_path)),img)
    print("mIou:{:.4f} mDice:{:.4f}".format(iousum/counter, dicesumm/counter))