mport torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import os
import argparse
import time
import numpy as np

from network import SupConResNet50
from loss import SupConLoss
from data import data_loader_stage1
from optimizer import load_optimizer, get_lr_scheduler
from utils import AverageMeter, accuracy, save_checkpoint, initialize_dir, save_model

def arg_parser():
    parser = argparse.ArgumentParser(description='arguments for CIFAR10 baseline training')

    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for data')

    parser.add_argument('--augment', default='AutoAugment', type=str, help='method for data augmentation')

    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='decay rate of learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='scheduler of learning rate')
    parser.add_argument('--eta_min', default=0.1**3, type=float, help='eta_min for cosine optimizer')
    parser.add_argument('--gamma', default=0.96, type=float, help='gamma for exp lr scheduler')
    parser.add_argument('--tau', default=0.07, type=float, help='temperature parameter for SupConLoss')

    parser.add_argument('--checkpoint_freq', default=40, type=str, help='checkpoint frequency')
    parser.add_argument('--print_freq', default=10, type=str, help='print frequency')
    parser.add_argument('--log_dir', default='./tensorboard/log', type=str, help='directory for log file')
    parser.add_argument('--load_pth', default='', type=str, help='directory for loading model')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='default device for running torch')

    args = parser.parse_args()

    return args

def load_model(args):
    model = SupConResNet50()

    if args.device == 'cuda':
        model = model.cuda()
        #model = nn.DataParallel(model)
        cudnn.benchmark = True

    if args.load_pth != '':
        #pth = torch.load('model_trained_embed.pth')
        pth = torch.load(args.load_pth)
        state_dict = pth['model']
        model.load_state_dict(state_dict)

    return model

def load_loss(args):
    return SupConLoss(tau=args.tau).cuda() if args.device == 'cuda' else SupConLoss(tau=args.tau)
    #return nn.CrossEntropyLoss()

def train(train_loader, model, criterion, optimizer, epoch, args):
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    model.train()

    end = time.time()

    for i, (x, y) in enumerate(train_loader):
        data_time.update(time.time() - end)

        dim = [2*y.shape[0]]
        dim.extend(list(x[0].shape[1:]))
        tmp = torch.zeros(tuple(dim))
        tmp[::2] = x[0].clone()
        tmp[1::2] = x[1].clone()
        x = tmp.float()

        x = x.to(args.device)
        y = y.to(args.device)

        feature = model(x)
        #feature = feature.view(y.shape[0], 2, -1)
        loss = criterion(feature, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), y.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # Print informations
        if (i + 1) % args.print_freq == 0:
            print(f'Train: {epoch+1} '+
                    f'iterate: [{i+1}/{len(train_loader)}] '
                    f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    f'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    f'Lr: {optimizer.param_groups[0]["lr"]:.6f} '
                    f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})')

    return train_loss.avg

if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)

    args = arg_parser()
    
    # initialize checkpoint directory
    initialize_dir("./checkpoint")

    # data loader
    train_loader = data_loader_stage1(args)

    # model + loss
    model = load_model(args)
    criterion = load_loss(args)

    # optimizer + lr scheduler
    optimizer = load_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)

    # tensorboardX
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(args.log_dir)

    # summary
    summary(model, input_size=(3, 32, 32))
    #input()

    end = time.time()

    for epoch in range(args.epochs):

        # train
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        print('[Train] epoch:', epoch+1)
        print('[Train] train_loss:', train_loss)

        # tensorboardX
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch+1)
        summary_writer.add_scalar('train_loss', train_loss, epoch+1)
        summary_writer.add_text('lr', str(optimizer.param_groups[0]['lr']), epoch+1)
        summary_writer.add_text('train_loss', str(train_loss), epoch+1)

        if epoch % args.checkpoint_freq == 0:
            save_model(model, optimizer, epoch+1, './checkpoint/' + f'ckpt_epoch_{epoch+1}.pth', args)

        lr_scheduler.step()

    print(f"Total Elapsed time: {time.time() - end}")

    save_model(model, optimizer, args.epochs, 'model_trained_embed.pth', args)

    summary_writer.close()
