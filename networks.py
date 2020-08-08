#!/usr/bin/env python3.6


import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from functools import partial

from layers import upSampleConv, convBatch, residualConv


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class UNet(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()
        self.name = "UNet"

        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))

        self.bridge = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                    residualConv(nG * 8, nG * 8),
                                    convBatch(nG * 8, nG * 8))

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        input = input.float()
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        bridge = self.bridge(x2)

        y0 = self.deconv1(bridge)
        y1 = self.deconv2(self.conv5(torch.cat((y0, x2), dim=1)))
        y2 = self.deconv3(self.conv6(torch.cat((y1, x1), dim=1)))
        y3 = self.conv7(torch.cat((y2, x0), dim=1))

        return self.final(y3)


class DeepMedic(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.name = "DeepMedic"
        self.path1 = nn.Sequential(convBatch(nin, 30, padding=0), convBatch(30, 30, padding=0),
                                    convBatch(30, 40, padding=0), convBatch(40, 40, padding=0), convBatch(40, 40, padding=0), convBatch(40, 40, padding=0),
                                    convBatch(40, 50, padding=0), convBatch(50, 50, padding=0))

        self.path2 = nn.Sequential(convBatch(nin, 30, padding=0), convBatch(30, 30, padding=0),
                                    convBatch(30, 40, padding=0), convBatch(40, 40, padding=0), convBatch(40, 40, padding=0), convBatch(40, 40, padding=0),
                                    convBatch(40, 50, padding=0), convBatch(50, 50, padding=0))

        self.upsample = partial(F.interpolate, scale_factor=3, mode='nearest')

        self.fullyconv = nn.Sequential(convBatch(100, 150, kernel_size=1, padding=0), convBatch(150, 150, kernel_size=1, padding=0))
        self.final = nn.Conv2d(150, nout, kernel_size=1)

    def forward(self, input1, input2):
        input1 = input1.float()
        input2 = input2.float()
        path1 = self.path1(input1)
        path2 = self.upsample(self.path2(input2))

        together = torch.cat((path1, path2), dim=1)
        together = self.fullyconv(together)

        return self.final(together)
