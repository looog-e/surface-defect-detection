# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:27:08 2021

@author: 零
"""

import torch
import torch.nn as nn



class CBG_layer(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride, padding=0):
        super(CBG_layer, self).__init__()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size  = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.cbg_conv = nn.Sequential(
            nn.Conv2d(self.in_feature, self.out_feature, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_feature),
            nn.ReLU()
#             nn.GELU()
        )
        
    def forward(self, x):
        x = self.cbg_conv(x)
        return x