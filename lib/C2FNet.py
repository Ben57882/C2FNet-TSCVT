"""

-*- coding:utf-8 -*-
author: SijieLiu
e-mail: sijieliu_123@sina.com
datetime:2022/01/01 19:31
software: PyCharm
Description: Improved C2FNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.Unet import UNet
import time
import torchvision.models as models
from utils.tensor_ops import cus_sample, upsample_add
from thop import clever_format
from thop import profile
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
class MSCA(nn.Module):
    def __init__(self, channels=32, r=2):
        super(MSCA, self).__init__()
        out_channels = int(channels//r)
        #local_att
        self.local_att =nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding= 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(channels)
        )

        #global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei
class ACFM(nn.Module):
    def __init__(self, channel=32):
        super(ACFM, self).__init__()

        self.msca = MSCA()
        self.upsample = cus_sample
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x, y, up=True):
        if up:
            y = self.upsample(y, scale_factor=2)
        # xy = torch.cat((x, y),1)
        xy = x + y
        wei = self.msca(xy)
        xo = x * wei + y * (1 - wei)
        xo = self.conv(xo)

        return xo
class DGCM(nn.Module):
    def __init__(self, channel=32):
        super(DGCM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)

        self.h2l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.h2h = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.mscah = MSCA()
        self.mscal = MSCA()

        self.upsample_add = upsample_add
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x):
        # first conv
        x_h = self.h2h(x)
        x_l = self.h2l(self.h2l_pool(x))
        x_h = x_h * self.mscah(x_h)
        x_l = x_l * self.mscal(x_l)
        out = self.upsample_add(x_l, x_h)
        # out = out + x_5
        out = self.conv(out)

        return out
class MBR(nn.Module):
    expansion =1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(MBR, self).__init__()
        # branch1
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample1 = upsample
        self.stride1 = stride
        # barch2
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv4 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,
                                   padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv_cat = BasicConv2d(2 * inplanes, inplanes, 3, padding=1)
        self.upsample2 = upsample
        self.stride2 = stride

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.upsample1 is not None:
            residual = self.upsample1(x)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        out2 = self.conv4(out2)
        out2 = self.bn4(out2)

        if self.upsample2 is not None:
            residual = self.upsample2(x)
        out = self.conv_cat(torch.cat((out1, out2), 1))
        out += residual
        out = self.relu(out)

        return out
class Refine(nn.Module):
    def __init__(self):
        super(Refine,self).__init__()
        self.upsample = cus_sample

    def forward(self, attention, x1, x2, x3):
        x1 = x1 + torch.mul(x1, self.upsample(attention, scale_factor=2))
        x2 = x2 + torch.mul(x2, self.upsample(attention, scale_factor=2))
        x3 = x3 + torch.mul(x3, attention)

        return x1, x2, x3
class C2FNet(nn.Module):
    def __init__(self, channel=32):
        super(C2FNet, self).__init__()

        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        #Decoder 1
        self.rfb0_1 = RFB_modified(64, channel)
        self.rfb1_1 = RFB_modified(256, channel)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        #Decoder 2
        self.rfb0_2 = RFB_modified(64, channel)
        self.rfb1_2 = RFB_modified(256, channel)
        self.rfb2_2 = RFB_modified(512, channel)

        self.acfm3 = ACFM()
        self.acfm2 = ACFM()
        self.dgcm3 = DGCM()
        self.dgcm2 = DGCM()

        self.upconv3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.classifier1 = nn.Conv2d(channel, 1, 1)

        self.refine = Refine()
        self.acfm1 = ACFM()
        self.acfm0 = ACFM()
        self.dgcm1 = DGCM()
        self.dgcm0 = DGCM()

        self.upconv1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv0 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.Unet = UNet()

        self.conv1 = BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.classifier2 = nn.Conv2d(channel // 2, 1, 1)
        self.upsample_add = upsample_add

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.uconv1 = BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.uconv2 = BasicConv2d(channel * 2, channel // 2, kernel_size=3, stride=1, padding=1, relu=True)
        # Components of CIM module4
        self.inplanes = 16
        self.deconv1 = self._make_CIM(MBR, 16, 3, stride=2)
        self.deconv2 = self._make_CIM(MBR, 16, 3, stride=2)
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        x2_1 = self.rfb2_1(x2)  # channel -> 32
        x3_1 = self.rfb3_1(x3)  # channel -> 32
        x4_1 = self.rfb4_1(x4)  # channel -> 32

        x43 = self.acfm3(x3_1, x4_1)
        out43 = self.upconv3(self.dgcm3(x43) + x43)

        x432 = self.acfm2(x2_1, out43)
        out432 = self.upconv2(self.dgcm2(x432) + x432)

        s2 = self.classifier1(out432)

        x0, x1, x2 = self.refine(s2.sigmoid(), x, x1, x2)

        x0_1 = self.rfb0_1(x0)
        x1_1 = self.rfb1_1(x1)
        x2_2 = self.rfb2_2(x2)

        x2 = self.up1(x2_2)
        x21 = torch.cat((x2, x1_1), 1)
        x21 = self.uconv1(x21)
        x210 = torch.cat((x21, x0_1), 1)

        x210 = self.uconv2(x210)

        s0 = self.deconv1(x210)
        s0 = self.deconv2(s0)
        s0 = self.classifier2(s0)
        s2 = F.interpolate(s2, scale_factor=8, mode='bilinear', align_corners=False)

        return s2, s0

    def _make_CIM(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            print("--------------")
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            print("*******************")
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
if __name__ == '__main__':
    torch.cuda.set_device(0)
    ras = C2FNet().cuda()
    input_tensor = torch.randn(2, 3, 352, 352).cuda()
    total_time = 0
    i=0
    torch.cuda.synchronize()
    start = time.time()
    out,_ = ras(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    single_fps = 1 / (end - start)
    total_time += end - start
    fps = (i + 1) / total_time
    print(' ({:.2f} fps total_time:{:.2f} single_fps:{})'.format(fps, total_time,single_fps))





    print(out[0].size())
    print(out[1].size())
    macs, params = profile(ras, inputs=(input_tensor,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

