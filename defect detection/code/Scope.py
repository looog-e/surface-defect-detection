# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:28:46 2021

@author: 零
"""

import torch
import torch.nn as nn
from CBG import CBG_layer


class Scope_layer(nn.Module):
    def __init__(self, n, in_feature, out_feature, kernel_size, stride, padding):
        super(Scope_layer, self).__init__()
        
        self.n = n
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cbg_conv = CBG_layer(self.in_feature, self.out_feature, self.kernel_size, self.stride, self.padding)
        
        self.scope_conv = nn.Sequential(*[self.cbg_conv for i in range(self.n)])
        
    def forward(self, x):
        for i, layer in enumerate(self.scope_conv):
            if i == 0:
                x = layer(x)
                x1 = x
            else:
                x1 = layer(x1)
                x = torch.cat([x, x1], dim=1)
        return x