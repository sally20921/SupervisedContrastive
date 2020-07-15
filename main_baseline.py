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

import network
from data import data_loader
from optimizer import *
from utils import AverageMeter, accuracy, save_checkpoint, save_model, initialize_dir

def arg_parser():
    parser = argparse.ArgumentParser(description='arguments for CIFAR10 baseline training')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=500, type=int, help='epochs')

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for data')

    parser.add_argument('--augment', default='AutoAugment', type=str, help='method for data augmentation')

    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.75, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='decay rate of learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='scheduler of learning rate')
    parser.add_argument('--eta_min', default=0.75**3, type=float, help='eta_min for cosine optimizer')

    parser.add_argument('--checkpoint_freq', default=40, type=str, help='checkpoint frequency')
    parser.add_argument('--print_freq', default=10, type=str, help='print frequency')
    parser.add_argument('--log_dir', default='./tensorboard/log_base', type=str, help='directory for log file')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='default device for running torch')

    args = parser.parse_args()

    return args

def load_model(args):
    model = network.ResNet50Base()

    if args.device == 'cuda':
        model = model.cuda()
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    return model

def load_loss(args):
    return nn.CrossEntropyLoss()

def train(train_loader, model, criterion, optimizer, epoch, args):
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    for i, (x, y) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
        top1.update(prec1[0], x.size(0))
        top5.update(prec5[0], x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # Print informations
        if (i + 1) % args.print_freq == 0:
            print(f'Train: {epoch+1} '+
                    f'iterate: [{i+1}/{len(train_loader)}] '
                    f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    f'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    f'Lr: {optimizer.param_groups[0]["lr"]:.6f} '
                    f'prec1: {top1.val:.3f} ({top1.avg:.3f}) '
                    f'prec5: {top5.val:.3f} ({top5.avg:.3f}) '
                    f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})')

    return train_loss.avg, top1.avg

def validate(val_loader, model, criterion, epoch, args):
    val_loss = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (x, y) in enumerate(val_loader):
            x = x.to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss = criterion(output, y)

            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            top1.update(prec1.item(), x.size(0))
            top5.update(prec5.item(), x.size(0))

            val_loss.update(loss.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # Print informations
            if (i + 1) % args.print_freq == 0:
                print(f'Val: [{epoch+1}] '
                        f'iterate: [{i+1}/{len(val_loader)}] '
                        f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        f'prec1: {top1.val:.3f} ({top1.avg:.3f}) '
                        f'prec5: {top5.val:.3f} ({top5.avg:.3f}) '
                        f'Loss: {val_loss.val:.4f} ({val_loss.avg:.4f})')

    return val_loss.avg, top1.avg

if __name__ == '__main__':
    args = arg_parser()
    
    best_acc = 0

    # initialize checkpoint directory
    initialize_dir("./checkpoint")

    # data loader
    train_loader, val_loader = data_loader(args)

    # model + loss
    model = load_model(args)
    criterion = load_loss(args)

    # optimizer + lr scheduler
    optimizer = load_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)

    # tensorboardX
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(args.log_dir)

    # summary model
    summary(model, input_size=(3, 32, 32))
    #input()

    end = time.time()

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)

        print('[Train] epoch:', epoch+1)
        print('[Train] train_loss:', train_loss)
        print('[Train] train_acc:', train_acc)

        # validate
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, args)

        print('[Validate] epoch:', epoch+1)
        print('[Validate] val_loss:', val_loss)
        print('[Validate] val_acc:', val_acc)

        # tensorboardX
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch+1)
        summary_writer.add_scalar('train_loss', train_loss, epoch+1)
        summary_writer.add_scalar('train_acc', train_acc, epoch+1)
        summary_writer.add_scalar('val_loss', val_loss, epoch+1)
        summary_writer.add_scalar('val_acc', val_acc, epoch+1)
        summary_writer.add_text('lr', str(optimizer.param_groups[0]['lr']), epoch+1)
        summary_writer.add_text('train_loss', str(train_loss), epoch+1)
        summary_writer.add_text('train_acc', str(train_acc), epoch+1)
        summary_writer.add_text('val_loss', str(val_loss), epoch+1)
        summary_writer.add_text('val_acc', str(val_acc), epoch+1)

        best_acc = max(best_acc, val_acc)

        if (epoch+1) % args.checkpoint_freq == 0:
            save_model(model, optimizer, epoch+1, './checkpoint/' + f'ckpt_epoch_{epoch+1}.pth', args)

        lr_scheduler.step()

    print(f"Total Elapsed time: {time.time() - end}")
    print('Best accuracy:', best_acc)

    save_model(model, optimizer, epoch+1, 'trained_final.pth', args)

    summary_writer.close()
