import argparse
import os
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.densenet_simclr import DenseNetSimCLR
from models.lenet_simclr import LeNetSimCLR
from models.resnet_simclr import ResNetSimCLR
from models.vgg_simclr import VggSimCLR
from utils import *
from simclr import SimCLR

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/datasets/', help='path to dataset')
parser.add_argument('--dataset-name', default='mnist', help='training dataset name',
                    choices=['mnist', 'mnist_raw', 'fashion_mnist', 'k_mnist', 'cifar10', 'cifar100', 'stl10',
                             'coco', 'tinyimagenet'])
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ["lenet5", "densenet40-12"]
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--pretrained', action='store_true', default=False, help='Use the pretrained cnn')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out-dim', default=128, type=int, help='feature dimension (default: 128)')
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for CL training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--layers', default=40, type=int, help='total number of DenseNet layers (default: 40)')
parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=1.0, type=float, help='compression rate in transition stage (default: 1.0)')
parser.add_argument('--no-bottleneck', dest='bottleneck', default=True, action='store_false', help='To not use bottleneck block')
parser.set_defaults(bottleneck=True)
parser.add_argument('--num-classes', default=10, type=int, help='total number of classes (default: 10)')
parser.add_argument('--save-dir', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/checkpoints/', help='path to save checkpoints')
parser.add_argument('--restore-file', default=None, help='filename from which to load checkpoint (default: <save-dir>/checkpoint_xxx.pth')
parser.add_argument('--optimizer', default='adam', help='optimizer name', choices=['SGD', 'Adam', 'Adadelta'])
parser.add_argument('--scheduler', default=None, help='scheduler name', choices=[None, 'CosineAnnealing', 'MultiStep', 'Exponential'])  # <fix>
parser.add_argument('--milestones', default='(150, 225)', metavar='B', help='the milestones for Scheduler MultiStepLR')
parser.add_argument('--gamma', default=0.1, type=float, help='the gama for Scheduler MultiStepLR')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--cl-model', default='SimCLR', help='the name of contrastive learning framework',
                    choices=['SimCLR', 'MoCo_V1', 'MoCo_V2', 'BYOL'])
parser.add_argument('--brightness', default=0.8, type=float, help='brightness value in ColorJitter')
parser.add_argument('--contrast', default=0.8, type=float, help='contrast value in ColorJitter')
parser.add_argument('--saturation', default=0.8, type=float, help='saturation value in ColorJitter')
parser.add_argument('--hue', default=0.2, type=float, help='hue value in ColorJitter')
parser.add_argument('--ResizedCropScale', default='(0.08, 1)', metavar='B', help='the scale for transforms.RandomResizedCrop')
parser.add_argument('--data-setup', default='mnist', help='the processed data setup',
                    choices=['none', 'mnist', 'cifar', 'coco', 'tinyimagenet'])
parser.add_argument('--claLoss-weight', default=1., type=float, metavar='D', help='weight for classification loss')
parser.add_argument('--conLoss-weight', default=1., type=float, metavar='D', help='weight for contrastive loss')


def load_model(args):
    if args.arch.startswith("lenet"):
        model = LeNetSimCLR(args.num_classes, args.out_dim)
    elif args.arch.startswith("densenet"):
        model = DenseNetSimCLR(args.layers, args.num_classes, args.growth, out_dim=args.out_dim,
                               reduction=args.reduce, bottleneck=args.bottleneck, dropRate=args.droprate)
    elif args.arch.startswith("vgg"):
        model = VggSimCLR(base_model=args.arch, pretrained=args.pretrained, num_classes=args.num_classes, out_dim=args.out_dim)
    elif args.arch.startswith("resnet"):
        model = ResNetSimCLR(base_model=args.arch, pretrained=args.pretrained, num_classes=args.num_classes, out_dim=args.out_dim)
    return model


def load_optimizer(args, model):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    return optimizer


def load_scheduler(args, optimizer):
    # Set the learning rate of each parameter group using a cosine annealing schedule
    if args.scheduler == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0003, last_epoch=-1)
    # decay the lr with scale value in fixed epoch
    elif args.scheduler == "MultiStep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(eval(args.milestones)),
                                                         gamma=args.gamma, last_epoch=-1)
    # exponentially decay the lr with scale 0.99 every epoch
    elif args.scheduler == "Exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
    return scheduler


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # use the cl-model type to choose the transformation type
    dataset = ContrastiveLearningDataset(args)

    # whether or not to transform the original image
    train_dataset = dataset.get_train_dataset(args.dataset_name, args.n_views, args.data_setup)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_dataset = dataset.get_val_dataset(args.dataset_name, args.n_views, args.data_setup)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)

    model = load_model(args)
    model.to(args.device)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    best_acc = 0
    if args.restore_file:
        if os.path.isfile(args.restore_file):
            print("=> loading checkpoint '{}'".format(args.restore_file))
            checkpoint = torch.load(args.restore_file)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['cla_acc'] + checkpoint['con_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.restore_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_file))

    optimizer = load_optimizer(args, model)
    scheduler = load_scheduler(args, optimizer)

    # Run
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, dataset_name=args.train_dataset_name, args=args)
    simclr.run(train_loader, val_loader, best_acc=best_acc)


if __name__ == "__main__":
    main()