from __future__ import print_function

import sys
import argparse
import time
import math

import torch

from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    opt = parser.parse_args()



def set_model(opt):
    model = SupConResNet(name=opt.name)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    return model, classifier, criterion 

def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    #one epoch training 
    model.eval()
    classifier.train()

def main():
    opt = parse_option()

    train_loader, val_loader = set_loader(opt)

    model, classifier, criterion = set_model(opt)

    for epoch in range(1, opt.epochs+1):
        #train for one epoch 
        loss = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)
        
        #eval for one epoch 
        loss =  validate(val_loader, model, classifier, criterion, opt)


if __name__=='__main__':
    main()
