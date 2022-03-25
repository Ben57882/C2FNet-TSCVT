import torch
torch.cuda.current_device()
torch.cuda._initialized=True
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
# from ..BBSC2F_P1.lib.BBS_C2F import BBS_C2FNet

from lib.C2FNet import C2FNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from utils.AdaX import AdaXW


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def LCE_loss(pred1, pred2, mask):
    loss1 = structure_loss(pred1, mask)
    loss2 = structure_loss(pred2, mask)
    loss = loss1 + loss2
    return loss

def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record3 = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        file = open("loss.txt", "a")
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            pred1, pred2 = model(images)
            # ---- loss function ----
            loss3 = LCE_loss(pred1, pred2, gts)
            loss = loss3
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record3.update(loss3.data, opt.batchsize)
        # ---- train visualization ----

        if i % 20 == 0 or i == total_step:
            file_name = 'lr_{}_train_results.txt'.format('1e-4')
            file = open(file_name, "a")
            test_result = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-3: {:.4f}]'.format(datetime.now(),epoch,opt.epoch, i,total_step,loss_record3.show())
            file.write(test_result + '\n')
            print(test_result)

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    visual = {"time": datetime.now(),
              "Epoch ": epoch,
              "loss": loss_record3.show()

              }
    file.write(str(visual) + '\n\n')
    if (epoch+1)  % 5 == 0:
        torch.save(model.state_dict(), save_path + 'C2FNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'C2FNet-%d.pth' % epoch)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=40, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='data/TrainDataset', help='path_to_train_dataset')
    parser.add_argument('--train_save', type=str,
                        default='C2FNet')
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = C2FNet().cuda()

    params = model.parameters()
    optimizer = AdaXW(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    print(image_root)
    print(gt_root)
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
