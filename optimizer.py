mport torch
import torch.optim as optim

def load_optimizer(args, model):
    if args.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, weight_decay=args.weight_decay, momentum=args.momentum, centered=False)
    else:
        raise ValueError("No default optimizer is set.")

def get_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    elif args.lr_scheduler == 'exp':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    else:
        raise ValueError("No such lr scheduler is set.")
