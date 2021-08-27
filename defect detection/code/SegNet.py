# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:31:02 2021

@author: 零
"""

class SegNet(nn.Module):
    def __init__(self, input_size, device):
        super(SegNet, self).__init__()
        self.input_size = input_size
        self.device = device
        
        self.cbg_conv1 = CBG_layer(3, 6, 3, 1, 1)
        self.scope_conv1 = Scope_layer(4, 6, 6, 3, 1, 1)
        
        self.cbg_conv2 = CBG_layer(3, 30, 1, 1)
        self.pool1 = nn.AvgPool2d((input_size[0], input_size[1]), (1, 1), count_include_pad=False)
        self.lbg_conv1 = LBG_layer(30, 5)
        self.lbg_conv2 = LBG_layer(5, 30)
        
        self.cbg_conv3 = CBG_layer(30, 32, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.res_conv1 = Res_layer(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.res_conv2 = Res_layer(64, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.res_conv3 = Res_layer(128, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.cbg_conv4 = CBG_layer(256, 256, 5, 1, 2)
        
        self.d_conv1 = DBG_layer(256, 128, 3, 2, 1, 1)
        self.cbg_conv5 = CBG_layer(256, 256, 5, 1, 2)
        self.d_conv2 = DBG_layer(256, 64, 3, 2, 1, 1)
        self.cbg_conv6 = CBG_layer(128, 128, 5, 1, 2)
        self.d_conv3 = DBG_layer(128, 32, 3, 2, 1, 1)
        self.cbg_conv7 = CBG_layer(64, 64, 5, 1, 2)
        self.d_conv4 = DBG_layer(64, 16, 3, 2, 1, 1)
        
        self.cbg_conv8 = CBG_layer(46, 32, 3, 1, 1)
        self.cbg_conv9 = CBG_layer(32, 8, 3, 1, 1)

        
        self.pool6 = nn.AvgPool2d((int(input_size[0]/16), int(input_size[1]/16)), (1, 1), count_include_pad=False)
        self.lbg_conv3 = LBG_layer(256, 8)

        
        self.cbg_conv10 = CBG_layer(8, 8, 3, 1, 1)
        self.cbg_conv11 = CBG_layer(8, 2, 3, 1, 1)
 
    def forward(self, x):
        
        x1 = self.cbg_conv1(x)
        x2 = self.scope_conv1(x1)
        x1 = torch.cat([x1, x2], dim=1)
        
        x = self.cbg_conv2(x)
        x = self.pool1(x).view(x.shape[0], x.shape[1])
        x = self.lbg_conv1(x)
        x = self.lbg_conv2(x)
        x = x1 * x.view(x.shape[0], x.shape[1], 1, 1)
    
        # encode part
        x0 = x
        x1 = self.cbg_conv3(x0)
        x1 = self.pool2(x1)
        x2 = self.res_conv1(x1)
        x2 = self.pool3(x2)
        x3 = self.res_conv2(x2)
        x3 = self.pool4(x3)
        x4 = self.res_conv3(x3)
        x4 = self.pool5(x4)
        x4 = self.cbg_conv4(x4)

        
        # decode part
        x = self.d_conv1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.cbg_conv5(x)
        x = self.d_conv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.cbg_conv6(x)
        x = self.d_conv3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.cbg_conv7(x)
        x = self.d_conv4(x)
        x = torch.cat([x, x0], dim=1)
        x = self.cbg_conv8(x)
        x = self.cbg_conv9(x)
        
        
        x4 = self.pool6(x4).view(x4.shape[0], x4.shape[1])
        x4 = self.lbg_conv3(x4)
        x = x * x4.view(x4.shape[0], x4.shape[1], 1, 1)
        x = self.cbg_conv10(x)
        x = self.cbg_conv11(x)
        x = nn.Softmax(dim=1)(x)

        return x