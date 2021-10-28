import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class NoamOpt:
    "Optimizer wrapper that implements rate decay (adapted from\
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

        self.state_dict = self.optimizer.state_dict()
        self.state_dict['step'] = 0
        self.state_dict['rate'] = 0

    def step(self):
        "Update parameters and rate"
        self.state_dict['step'] += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.state_dict['rate'] = rate
        self.optimizer.step()
        for k, v in self.optimizer.state_dict().items():
            self.state_dict[k] = v

    def rate(self, step=None):
        "Implement 'lrate' above"
        if step is None:
            step = self.state_dict['step']
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

class AdamOpt:
    "Adam optimizer wrapper"
    def __init__(self, params, lr, optimizer):
        self.optimizer = optimizer(params, lr)
        self.state_dict = self.optimizer.state_dict()

    def step(self):
        self.optimizer.step()
        self.state_dict = self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict



class WarmupLRScheduler(torch.optim.lr_scheduler._LRScheduler):
        """
        Warmup learning rate until `total_steps`

        Args:
            optimizer (Optimizer): wrapped optimizer.
            configs (DictConfig): configuration set.
        """
        def __init__(
                self,
                model_size,
                factor,
                warmup,
                optimizer,
        ) -> None:
            warmup_steps = 2
            peak_lr = 0.1
            self.init_lr = 0.001
            warmup_steps = 2

            if warmup_steps != 0:
                warmup_rate = peak_lr - self.init_lr
                self.warmup_rate = warmup_rate / warmup_steps
            else:
                self.warmup_rate = 0
            self.update_steps = 1
            self.lr = self.init_lr
            self.warmup_steps = 2
            super().__init__(optimizer)

        def set_lr(self, optimizer, lr):
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        def step(self, val_loss = None):
            #print(self.lr)
            if self.update_steps < self.warmup_steps:
                lr = self.init_lr + self.warmup_rate * self.update_steps
                self.set_lr(self.optimizer, lr)
                self.lr = lr
            self.update_steps += 1
            return self.lr

        def rate(self, step=None):
            "Implement 'lrate' above"
            if step is None:
                step = self.step
            return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


