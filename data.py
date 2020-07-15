mport torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from auto_augment import AutoAugment

def get_mean_std():
    data = torchvision.datasets.CIFAR10(
            root='./dataset', train=True, download=True)
    x = np.concatenate([np.asarray(data[i][0]) for i in range(len(data))])
    mean = np.mean(x, axis=(0,1)) / 255
    std = np.std(x, axis=(0,1)) / 255
    return mean, std

def data_loader(args):
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    else:
        raise ValueError('Unavailable dataset "%s"' % (dataset))
    
    transform_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ]

    if args.augment == 'AutoAugment':
        transform_train.append(AutoAugment())
    elif args.augment == 'Basic':
        transform_train.extend([
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
                ], 0.8),
            transforms.RandomGrayscale(0.1),
            ])
    else:
        raise ValueError('No such augmentation policy is set!')

    transform_train.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    transform_train = transforms.Compose(transform_train)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
                root='./dataset', train=True, download=True, transform=transform_train)
        val_set = torchvision.datasets.CIFAR10(
                root='./dataset', train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader

def data_loader_stage1(args):
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    else:
        raise ValueError('Unavailable dataset "%s"' % (dataset))
    
    transform_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ]

    if args.augment == 'AutoAugment':
        transform_train.append(AutoAugment())
    elif args.augment == 'Basic':
        transform_train.extend([
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
                ], 0.8),
            transforms.RandomGrayScale(0.1),
            ])
    else:
        raise ValueError('No such augmentation policy is set!')

    transform_train.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    transform_train = DuplicateTransform(transforms.Compose(transform_train))

    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
                root='./dataset', train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    return train_loader

class DuplicateTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return [self.transform(sample)]*2
