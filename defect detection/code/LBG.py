# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:25:21 2021

@author: 零
"""

class LBG_layer(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(LBG_layer, self).__init__()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.lbg_conv = nn.Sequential(
            nn.Linear(self.in_feature, self.out_feature),
            nn.BatchNorm1d(self.out_feature),
            nn.ReLU()
#             nn.GELU()
        )
        
    def forward(self, x):
        x = self.lbg_conv(x)
        return x