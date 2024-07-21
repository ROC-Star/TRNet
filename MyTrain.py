"""
@File: Net.py
@Time: 2023/11/30
@Author: rp
@Software: PyCharm

"""

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print('USE GPU 1')

import torch
import torch.nn.functional as F


import numpy as np
from datetime import datetime


from torch import nn, optim
from torchvision.utils import make_grid
from tools.data import get_loader, test_dataset
from tools.tools import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from lib.Net import Net


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, )
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, depth) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()

            s, s1, s2, s3, s4 = model(images, depth)

            sal_loss = structure_loss(s, gts)
            sal_loss1 = structure_loss(s1, gts)
            sal_loss2 = structure_loss(s2, gts)
            sal_loss3 = structure_loss(s3, gts)
            sal_loss4 = structure_loss(s4, gts)

            loss = sal_loss + sal_loss1 + sal_loss2 + sal_loss3 + sal_loss4
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            if i % 100 == 0 or i == total_step or i == 1:
                # loop.set_description(f'Epoch [{epoch}]')
                # loop.set_postfix(salloss=sal_loss.data,edge_loss=edge_loss.data,salloss1=sal_loss1.data)
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f} loss:{:4f} ||sal_loss1:{:4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           optimizer.state_dict()['param_groups'][0]['lr'], loss.data, sal_loss1.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, loss:{:4f} ||sal_loss1:{:4f}'.
                    format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data,
                           sal_loss1.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')
        # sal_loss_all /= epoch_step
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'CatNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'CatNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()
            res, _, _, _, _ = model(image, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'CatNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='./Datasets/RGBD_SOD/train_2985/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./Datasets/RGBD_SOD/TestDataset/NLPR/',
                        help='the test gt images root')
    parser.add_argument('--save_path', type=str, default='./snapshot/Exp1/', help='the path to save model and log')
    opt = parser.parse_args()


    cudnn.benchmark = True

    # build the model
    model = Net().cuda()
    # model.cuda()
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)


    optimizer = torch.optim.Adam(model.parameters(), opt.lr)


    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              depth_root=opt.train_root + 'Depth/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize)
    val_loader = test_dataset(image_root=opt.val_root + 'RGB/',
                              depth_root=opt.val_root + 'depth/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_mae = 1
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print(">>> start train...")
    for epoch in range(1, opt.epoch):
        # schedule
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        # train
        train(train_loader, model, optimizer, epoch, save_path)
        val(val_loader, model, epoch, save_path)
