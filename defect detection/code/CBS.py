# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:27:41 2021

@author: 零
"""

class CBS_layer(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride, padding=0):
        super(CBS_layer, self).__init__()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size  = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.cbs_conv = nn.Sequential(
            nn.Conv2d(self.in_feature, self.out_feature, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_feature),
            nn.Sigmoid()
#             nn.GELU()
        )
        
    def forward(self, x):
        x = self.cbs_conv(x)
        return x