# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:32:34 2021

@author: 零
"""

class evaluator():
    def __init__(self, outputs, labels):
        self.outputs = outputs
        self.labels = labels
        self.shape = self.outputs.shape
    
    def loss_fn(self):
        loss_value = nn.BCELoss()(self.outputs, self.labels)
        return loss_value
    
    def acc_fn(self):
        self.outputs = self.outputs.cpu()
        self.labels = self.labels.cpu()
        self.outputs[self.outputs >= 0.5] = 1.
        self.outputs[self.outputs < 0.5] = 0. 
        acc = (torch.sum(self.outputs == self.labels).item() / (self.shape[0]*self.shape[1]*self.shape[2]*self.shape[3]))
        acc_mIou = 0.
        acc_miou_p = 0.
        acc_miou_n = 0.

        for i in np.unique(self.labels.detach().numpy()):
            for j in range(self.shape[0]):
                acc_mIou += ((torch.sum((self.outputs[j, 0, :, :] == self.labels[j, 0, :, :]) & (self.labels[j, 0, :, :] ==i)).item() + 1) / (1 + torch.sum(self.outputs[j, 0, :, :] == i) + torch.sum(self.labels[j, 0, :, :] == i) - torch.sum((self.outputs[j, 0, :, :] == self.labels[j, 0, :, :]) & (self.labels[j, 0, :, :] == i))).item())
                acc_miou_n += ((torch.sum((self.outputs[j, 0, :, :] == self.labels[j, 0, :, :]) & (self.labels[j, 0, :, :] ==0)).item() + 1) / (1 + torch.sum(self.outputs[j, 0, :, :] == 0) + torch.sum(self.labels[j, 0, :, :] == 0) - torch.sum((self.outputs[j, 0, :, :] == self.labels[j, 0, :, :]) & (self.labels[j, 0, :, :] == 0))).item())
                acc_miou_p += ((torch.sum((self.outputs[j, 0, :, :] == self.labels[j, 0, :, :]) & (self.labels[j, 0, :, :] ==1)).item() + 1) / (1 + torch.sum(self.outputs[j, 0, :, :] == 1) + torch.sum(self.labels[j, 0, :, :] == 1) - torch.sum((self.outputs[j, 0, :, :] == self.labels[j, 0, :, :]) & (self.labels[j, 0, :, :] == 1))).item())
        acc_mIou /= (np.unique(self.labels.detach().numpy()).shape[0] * self.shape[0])
        acc_miou_p  /= (np.unique(self.labels.detach().numpy()).shape[0] * self.shape[0])
        acc_miou_n  /= (np.unique(self.labels.detach().numpy()).shape[0] * self.shape[0])
        return acc, acc_mIou, acc_miou_p, acc_miou_n