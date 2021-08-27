# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:29:40 2021

@author: 零
"""

class Res_layer(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride, padding):
        super(Res_layer, self).__init__()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.cbg_conv1 = CBG_layer(self.in_feature, int(self.in_feature/2), 1, 1)
        self.cbg_conv2 = CBG_layer(int(self.in_feature/2), int(self.in_feature/2), self.kernel_size,  1, self.padding)
        self.cbg_conv3 = CBG_layer(int(self.in_feature/2), self.out_feature-self.in_feature, 1, 1)
        
    def forward(self, x):
        x1 = self.cbg_conv1(x)
        x1 = self.cbg_conv2(x1)
        x1 = self.cbg_conv3(x1)
        x = torch.cat([x, x1], dim=1)
        return x