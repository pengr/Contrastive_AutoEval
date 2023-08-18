import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.densenet_simclr import DenseNetSimCLR
from models.lenet_simclr import LeNetSimCLR
from models.resnet_simclr import ResNetSimCLR
from models.vgg_simclr import VggSimCLR
from simclr import SimCLR
import os
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/datasets/', help='path to dataset')
parser.add_argument('--test-dataset-name', default='mnist', help='Unseen target dataset name',
                    choices=['svhn', 'usps', 'cifar10_1', 'cifar10_c', 'cifar100',
                             'cifar100_c', 'caltech', 'pascal','imagenet', 'tinyimagenet_c'])
parser.add_argument('--meta-dataset-name', default='mnist',
                    help='Meta-set dataset name', choices=['mnist', 'cifar10', 'cifar100', 'coco', 'tinyimagenet'])
parser.add_argument('--metaset-dir', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/metasets/MNIST',
                    help='path to save the generated meta-set')
parser.add_argument('--metaset-numLim', default=1e6, type=int, metavar='N', help='the range of selected meta-set')
model_names = sorted(name for name in models.__dict__  if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ["lenet", "densenet40-12"]
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--pretrained', action='store_true', default=False, help='Use the pretrained cnn')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
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

    model = load_model(args)
    model.to(args.device)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.restore_file:
        if os.path.isfile(args.restore_file):
            print("=> loading checkpoint '{}'".format(args.restore_file))
            checkpoint = torch.load(args.restore_file)
            args.start_epoch = checkpoint['epoch']
            cla_acc = checkpoint['cla_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.restore_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_file))

    simclr = SimCLR(model=model, optimizer=None, scheduler=None, dataset_name=args.test_dataset_name, args=args)

    epoch = checkpoint['epoch']
    print(f'\nEpoch:{epoch}')
    print(f'\nTest batch size:{args.test_batch_size}')

    # eval on unseen test set
    print(f'Test on {args.test_dataset_name}')
    test_con_acc = test_cla_acc = 0
    dataset = ContrastiveLearningDataset(args)  # use the cl-model type to choose the transformation type
    if args.test_dataset_name == 'tinyimagenet_c':
        corrupution_types = os.listdir(args.data+ "Tiny-ImageNet-C")
        for i in range(0, len(corrupution_types)):  # each corrupution_types
            test_con_acc_i = test_cla_acc_i = 0
            for j in range(1, 6):
                test_dataset = dataset.get_test_dataset(os.path.join(args.data, "Tiny-ImageNet-C", corrupution_types[i], f"{j}") , args.test_dataset_name,
                                                        args.n_views, args.data_setup, train_trans=False)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=args.test_batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)

                # evaluate on semantic classification and contrastive learning (Calculate both acc)
                test_con_acc_ij, test_cla_acc_ij = simclr.test(test_loader)
                test_con_acc_i += test_con_acc_ij
                test_cla_acc_i += test_cla_acc_ij
            test_con_acc_i /= 5
            test_cla_acc_i /= 5
            # print(f'\nSemantic classification accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_cla_acc_i))
            # print(f'Contrastive learning accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_con_acc_i))

            test_con_acc += test_con_acc_i
            test_cla_acc += test_cla_acc_i
        test_con_acc /= len(corrupution_types)
        test_cla_acc /= len(corrupution_types)
        ## predict for all corrupution_types
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))

    elif args.test_dataset_name == 'cifar10_c':
        corrupution_types = [os.path.splitext(i)[0] for i in os.listdir(args.data+"CIFAR-10-C")
                             if os.path.splitext(i)[0] != "labels" and os.path.splitext(i)[1] == ".npy"]
        for i in range(0, len(corrupution_types)):
            test_dataset = dataset.get_test_dataset(os.path.join(args.data, "CIFAR-10-C", corrupution_types[i]+".npy"), args.test_dataset_name,
                                                    args.n_views, args.data_setup, train_trans=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.test_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            # evaluate on semantic classification and contrastive learning (Calculate both acc)
            test_con_acc_i, test_cla_acc_i = simclr.test(test_loader)
            # print(f'\nSemantic classification accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_cla_acc_i))
            # print(f'Contrastive learning accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_con_acc_i))

            test_con_acc += test_con_acc_i
            test_cla_acc += test_cla_acc_i
        test_con_acc /= len(corrupution_types)
        test_cla_acc /= len(corrupution_types)
        ## predict for all corrupution_types
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))

    elif args.test_dataset_name == 'cifar100_c':
        corrupution_types = [os.path.splitext(i)[0] for i in os.listdir(args.data+"CIFAR-100-C")
                             if os.path.splitext(i)[0] != "labels" and os.path.splitext(i)[1] == ".npy"]
        for i in range(0, len(corrupution_types)):
            test_dataset = dataset.get_test_dataset(os.path.join(args.data, "CIFAR-100-C", corrupution_types[i]+".npy"), args.test_dataset_name,
                                                    args.n_views, args.data_setup, train_trans=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.test_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            # evaluate on semantic classification and contrastive learning (Calculate both acc)
            test_con_acc_i, test_cla_acc_i = simclr.test(test_loader)
            # print(f'\nSemantic classification accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_cla_acc_i))
            # print(f'Contrastive learning accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_con_acc_i))

            test_con_acc += test_con_acc_i
            test_cla_acc += test_cla_acc_i
        test_con_acc /= len(corrupution_types)
        test_cla_acc /= len(corrupution_types)
        ## predict for all corrupution_types
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))

    else:
        test_dataset = dataset.get_test_dataset(args.data, args.test_dataset_name, args.n_views, args.data_setup,
                                                train_trans=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        # evaluate on semantic classification and contrastive learning (Calculate both acc)
        test_con_acc, test_cla_acc = simclr.test(test_loader)
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))

    # load the regression model
    cla_acc = np.load(f'{args.save_dir}/accuracy_cla{epoch}.npy')
    con_acc = np.load(f'{args.save_dir}/accuracy_con{epoch}.npy')

    # the statistical correlation value
    rho, pval = stats.spearmanr(con_acc, cla_acc)
    print('\nSpearman\'s Rank correlation-rho', rho)
    print('Spearman\'s Rank correlation-pval', pval)
    rho, pval = stats.pearsonr(con_acc, cla_acc)
    print('\nPearsons correlation-rho', rho)
    print('Pearsons correlation-pval', pval)
    rho, pval = stats.kendalltau(con_acc, cla_acc)
    print('\nKendall\'s Rank correlation-rho', rho)
    print('Kendall\'s correlation-pval', pval)

    ## using regression model to predict preformance of unseen target sets
    slr = LinearRegression()
    slr.fit(np.array(con_acc.reshape(-1, 1)), np.array(cla_acc.reshape(-1, 1)))
    pred = slr.predict(np.array(test_con_acc).reshape(-1, 1))
    error = mean_squared_error(pred, np.array(test_cla_acc).reshape(-1, 1), squared=False)  # squared=False returns RMSE value
    print('\nLinear regression model predicts %4f, its absolute error is %4f' % (pred, error))

    robust_reg = HuberRegressor()
    robust_reg.fit(np.array(con_acc.reshape(-1, 1)), np.array(cla_acc.reshape(-1)))
    robust_pred = robust_reg.predict(np.array(test_con_acc).reshape(-1, 1))
    robust_error = mean_squared_error(robust_pred, np.array(test_cla_acc).reshape(-1, 1), squared=False)
    print('Robust Linear regression model predicts %4f, its absolute error is %4f' % (robust_pred, robust_error))


if __name__ == "__main__":
    main()