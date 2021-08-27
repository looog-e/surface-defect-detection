# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:26:51 2021

@author: 零
"""

import torch
import torch.nn as nn


class LBS_layer(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(LBS_layer, self).__init__()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.lbs_conv = nn.Sequential(
            nn.Linear(self.in_feature, self.out_feature),
            nn.BatchNorm1d(self.out_feature),
            nn.Sigmoid()
#             nn.GELU()
        )
        
    def forward(self, x):
        x = self.lbs_conv(x)
        return x