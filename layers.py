#!/usr/bin/env python3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Only for 0 <= se_size <= 2*pi
#Will only yield values between -1 and 1!
def dilatedSin(input, se_size = np.pi/4.0, left_clamp_offset = 0, right_clamp_offset = 0):
    input = torch.add(input, -se_size/2.0) #Adjust zero-crossing
    input_clamped = torch.clamp(input, min=-np.pi/2.0 + left_clamp_offset, max=np.pi/2.0 - se_size / 2 - right_clamp_offset)
    return torch.sin(input_clamped + se_size/2.0)


#Only for 0 <= se_size <= 2*pi
#Will only yield values between -1 and 1!
#Buggy?
def dilatedSinMemoryEfficient(input, se_size = np.pi/4.0, left_clamp_offset = 0, right_clamp_offset = 0):
    input.add_(-se_size/2.0) #Adjust zero-crossing
    input.clamp_(min=-np.pi/2.0 + left_clamp_offset, max=np.pi/2.0 - se_size/2 - right_clamp_offset)
    return torch.sin(input + se_size/2.0)


#Only for 0 <= se_size <= 2*pi
def slopedDilatedSin(input, se_size = np.pi/4.0, coeff = 0.1, left_clamp_offset = 0, right_clamp_offset = 0):
    val = dilatedSin(input, se_size, left_clamp_offset, right_clamp_offset)
    return val + torch.mul(input, coeff)

#Only for 0 <= se_size <= 2*pi
def clampedSlopedDilatedSin(input, se_size = np.pi/4.0, coeff = 0.1, left_clamp_offset = 0, right_clamp_offset = 0):
    val = dilatedSin(input, se_size, left_clamp_offset, right_clamp_offset)
    input_clamped = torch.clamp(input, min=0)
    return val + torch.mul(input_clamped, coeff)


class DilatedSin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return dilatedSin(input, np.pi/2.0)

class SlopedDilatedSin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return slopedDilatedSin(input, np.pi/2.0, 0.05)

class ClampedSlopedDilatedSin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return clampedSlopedDilatedSin(input, np.pi/2.0, 0.05)

def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        #DilatedSin()
        # SlopedDilatedSin()
        ClampedSlopedDilatedSin()
        # nn.PReLU()
    )

#
# def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
#     return nn.Sequential(
#         layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
#         nn.BatchNorm2d(nout),
#         nn.PReLU()
#     )


def upSampleConv(nin, nout, kernel_size=3, upscale=2, padding=1, bias=False):
    return nn.Sequential(
        nn.Upsample(scale_factor=upscale),
        convBatch(nin, nout, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
        convBatch(nout, nout, kernel_size=3, stride=1, padding=1, bias=bias),
    )


class residualConv(nn.Module):
    def __init__(self, nin, nout):
        super(residualConv, self).__init__()
        self.convs = nn.Sequential(
            convBatch(nin, nout),
            nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nout)
        )
        self.res = nn.Sequential()
        if nin != nout:
            self.res = nn.Sequential(
                nn.Conv2d(nin, nout, kernel_size=1, bias=False),
                nn.BatchNorm2d(nout)
            )

    def forward(self, input):
        out = self.convs(input)
        return F.leaky_relu(out + self.res(input), 0.2)
