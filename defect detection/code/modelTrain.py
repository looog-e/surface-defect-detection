# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:32:55 2021

@author: 零
"""

import copy
import torch
import numpy as np
import torch.nn as nn


class model_train():
    def __init__(self, model, dataloader, evaluator, optimizer, num_epoch, state, device):
        self.model = model 
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.state = state
        self.device = device
        self.running_loss = 0.
        self.running_acc = 0.
        self.running_acc_mIou = 0.
        self.running_acc_miou_p = 0.
        self.running_acc_miou_n = 0.
        self.best_acc = 0.
        self.acc_history_s = []
        self.acc_history = []
        self.acc_mIou_history_s = []
        self.acc_mIou_history = []
        
    def train(self):
        save_num = 1
        for epoch in range(self.num_epoch):
            for state in self.state:
                self.running_loss = 0.
                self.running_acc = 0.
                self.running_acc_mIou = 0.
                self.running_acc_miou_p = 0.
                self.running_acc_miou_n = 0.
                if state == "train":
                    self.model.train()
                else:
                    self.model.eval()
                for inputs, labels in self.dataloader[state]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with torch.autograd.set_grad_enabled(state=="train"):
                        outputs = self.model(inputs)
                        loss = self.evaluator(outputs, labels).loss_fn()
                        acc, acc_mIou, acc_miou_p, acc_miou_n = self.evaluator(outputs, labels).acc_fn()
                    if state == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    self.running_loss += loss.item() * inputs.size(0)
                    self.running_acc += acc * inputs.size(0)
                    self.running_acc_mIou += acc_mIou * inputs.size(0)
                    self.running_acc_miou_p += acc_miou_p * inputs.size(0)
                    self.running_acc_miou_n += acc_miou_n * inputs.size(0)
                    self.acc_history_s.append(self.running_acc)
                    self.acc_mIou_history_s.append(self.running_acc_mIou)
                epoch_loss = self.running_loss / len(self.dataloader[state].dataset)
                epoch_acc = self.running_acc / len(self.dataloader[state].dataset)
                epoch_acc_mIou = self.running_acc_mIou / len(self.dataloader[state].dataset)
                epoch_acc_miou_p = self.running_acc_miou_p / len(self.dataloader[state].dataset)
                epoch_acc_miou_n = self.running_acc_miou_n / len(self.dataloader[state].dataset) 
                print("Epoch {}/{} __ Phase {} loss: {:.4f}, acc: {:.4f}, acc_mIou: {:.4f}, acc_miou_p: {:.4f}, acc_miou_n: {:.4f}". format(epoch+1, self.num_epoch, state, epoch_loss, epoch_acc, epoch_acc_mIou, epoch_acc_miou_p, epoch_acc_miou_n))
                
                if state == "eval":
                    print("{}".format("_"*45))
                    self.acc_history.append(epoch_acc)
                    self.acc_mIou_history.append(epoch_acc_mIou)
                    if epoch_acc_miou_p > self.best_acc:
                        self.best_acc = epoch_acc_miou_p
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                    save_num += 1
        self.model.load_state_dict(best_model_wts)
        return self.model, (self.acc_history, self.acc_mIou_history, self.acc_history_s, self.acc_mIou_history_s)