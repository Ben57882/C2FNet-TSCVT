import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy import misc
from lib.C2FNet import C2FNet
from utils.dataloader import test_dataset
from PIL import Image
import cv2

import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy import misc
from lib.C2FNet import C2FNet
from utils.dataloader import test_dataset
import time
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='checkpoints/C2FNet/C2FNet-49.pth')

for _data_name in ['CAMO','CHAMELEON','COD10K']: #'CAMO','CHAMELEON','COD10K'
    data_path = 'data/TestDataset/{}'.format(_data_name)
    save_path = 'results/C2FNet49/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = C2FNet()
    #model = torch.nn.DataParallel(model)
    #torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    model.load_state_dict(torch.load(opt.pth_path), strict='False')
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        total_time = 0
        i = 0
        torch.cuda.synchronize()
        start = time.time()
        _, res = model(image)
        torch.cuda.synchronize()
        end = time.time()
        single_fps = 1 / (end - start)
        total_time += end - start
        fps = (i + 1) / total_time
        print(' ({:.2f} fps total_time:{:.2f} single_fps:{})'.format(fps, total_time, single_fps))


        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # print(np.shape(res))
        filepath, fullflname = os.path.split(name)
        print('save img to: ', save_path + fullflname)
        res = res * 255
        res = Image.fromarray(res.astype(np.uint8))
        res.save(save_path + fullflname)