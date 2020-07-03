from __future__ import print_function

import os
import sys
import argparse
import time
import math 

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

def main():
    opt =  parse_option()

    train_loader =  set_loader(opt)

    model, criterion = set_model(opt)

    optimizer = set_optimizer(opt, model)

    #training routine
    for epoch in range(1,  opt.epochs+1):
        adjust_learning_rate(opt, optimizer, epoch)

        loss = train(train_loader, model, criterion, optimizer,  epoch, opt)

if __name__=='__main__':
    main()
