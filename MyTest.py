"""
@File: Net.py
@Time: 2023/11/30
@Author: rp
@Software: PyCharm

"""

import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('USE GPU 0')

import cv2
import torch
import argparse
import numpy as np

import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from lib.Net import Net
from tools.data import test_dataset as EvalDataset


def evaluator(model, val_root, map_save_path, trainsize=352):
    val_loader = EvalDataset(image_root=val_root + 'RGB/',
                             depth_root=val_root + 'depth/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)

    model.eval()
    total = []
    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, depth, name, image_for_post = val_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            begin = time.time()
            output, _, _, _, _ = model(image, depth)
            end = time.time()
            t = end - begin
            total.append(t)
            output = F.upsample(output, size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            cv2.imwrite(map_save_path + name, output * 255)
            print('>>> saving prediction at: {}'.format(map_save_path + name))
        fps = val_loader.size / np.sum(total)
        print("fps:", fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap_path', type=str,
                        default='./snapshot/Exp1/CatNet_epoch_215.pth',
                        help='train use gpu')
    # parser.add_argument('--gpu_id', type=str, default='0',
    #                     help='train use gpu')
    parser.add_argument('--test_size', type=int, default=384,
                        help='training dataset size')
    opt = parser.parse_args()

    txt_save_path = './result1/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    cudnn.benchmark = True
    model = Net().cuda()

    # TODO: remove FC layers from snapshots
    model.load_state_dict(torch.load(opt.snap_path))
    model.eval()
    # 'CAMO', 'COD10K', 'NC4K', 'CHAMELEON'
    for data_name in ['SSD', 'SIP', 'NJU2K', 'NLPR', 'STERE', 'DES', 'LFSD', 'RGBD135', 'DUT-RGBD']:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root='./Datasets/RGBD_SOD/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=opt.test_size
        )
