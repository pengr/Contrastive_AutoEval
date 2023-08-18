import os
import shutil

import torch
import yaml

import numpy as np
import torch.utils.data as data
from PIL import Image
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
import random


# <<Are labels always...>> load from the given numpy file and process into RGB format
class MyMNIST(data.Dataset):
    """`mnist <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``mnist/processed/training.pt``
            and  ``mnist/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, sample_file, label_file, transform=None, target_transform=None):
        super(MyMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(str(root) + '//' + sample_file)
        self.targets = np.load(str(root) + '//' + label_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if len(self.data.shape) > 3:
            img, target = self.data[:, :, :, index], int(self.targets[index])
        else:
            img, target = self.data[:, :, index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')
        img = Image.fromarray(np.uint8(img)).convert('RGB')  # unit8

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if len(self.data.shape) > 3:
            return self.data.shape[3]
        else:
            return self.data.shape[2]


# <<Are labels always...>> load from the given numpy file
class MNIST_bg(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, sample_file, label_file, transform=None, target_transform=None):
        super(MNIST_bg, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(str(root) + '//' + sample_file)
        self.targets = np.load(str(root) + '//' + label_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[:, :, :, index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')
        img = Image.fromarray(np.uint8(img)).convert('L')  # unit8

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[3]


# Override the path and 'L' model of torchvision MNIST
class MyMNISTRAW(MNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)  #

    @property  # fix the MNIST path: xxx/MNIST/raw
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property  # fix the MNIST path: xxx/MNIST/processed
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy()).convert('RGB')  # original code uses mode='L'

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# Override the path and 'L' model of torchvision FashionMNIST
class MyFashionMNIST(FashionMNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)

    @property  # fix the FashionMNIST path: xxx/FashionMNIST/raw
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'FashionMNIST', 'raw')

    @property  # fix the FashionMNIST path: xxx/FashionMNIST/processed
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'FashionMNIST', 'processed')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy()).convert('RGB')  # original code uses mode='L'

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# Override the path and 'L' model of torchvision KMNIST
class MyKMNIST(KMNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)

    @property  # fix the KMNIST path: xxx/KMNIST/raw
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'KMNIST', 'raw')

    @property  # fix the KMNIST path: xxx/KMNIST/processed
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'KMNIST', 'processed')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy()).convert('RGB')  # original code uses mode='L'

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# copy from the rotation code
class CIFAR10_1(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, version='v6'):
        self.data = np.load('%s/cifar10.1_%s_data.npy' % (root, version))
        self.targets = np.load('%s/cifar10.1_%s_labels.npy' % (root, version)).astype('long')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)


class CIFAR10_c(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.data = np.load(root)
        self.targets = np.load('%s/labels.npy' % (os.path.dirname(root))).astype('long')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)


class CIFAR100_c(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.data = np.load(root)
        self.targets = np.load('%s/labels.npy' % (os.path.dirname(root))).astype('long')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)


class MyCIFAR10(data.Dataset):
    """`mnist <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``mnist/processed/training.pt``
            and  ``mnist/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, sample_file, label_file, transform=None, target_transform=None):
        super(MyCIFAR10, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(str(root) + '//' + sample_file)
        self.targets = np.load(str(root) + '//' + label_file).astype('long')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


class MyCIFAR100(data.Dataset):
    """`mnist <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``mnist/processed/training.pt``
            and  ``mnist/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, sample_file, label_file, transform=None, target_transform=None):
        super(MyCIFAR100, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(str(root) + '//' + sample_file)
        self.targets = np.load(str(root) + '//' + label_file).astype('long')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


class MyCoco(data.Dataset):

    labels_map = {
        'airplane': 0,    # train: 1680, val: 553
        'aeroplane': 0,
        'bicycle': 1,     # train: 929,  val: 428
        'bird': 2,        # train: 959,  val: 451
        'boat': 3,        # train: 888,  val: 475
        'bottle': 4,      # train: 377,  val: 167
        'bus': 5,         # train: 2000, val: 995
        'car': 6,         # train: 1761, val: 886
        'dog': 7,         # train: 1977, val: 1052
        'horse': 8,       # train: 1834, val: 867
        'motorcycle': 9,  # train: 2000, val: 1079
        'motorbike': 9,
        'person': 10,     # train: 2000, val: 2000
        'tv': 11,         # train: 1400, val: 688
        'tvmonitor': 11,
    }

    # Way 1: Convert the images in train, val, seed, unseen test sets of coco to numpy format
    # def __init__(self, root, transform=None, target_transform=None) -> None:
    #     super(MyCoco, self).__init__()
    #     self.transform = transform
    #     self.target_transform = target_transform
    #
    #     self.data = []
    #     self.targets = []
    #
    #     for class_folder in os.listdir(root):
    #         class_folder_path = os.path.join(root, class_folder)
    #         class_files = os.listdir(class_folder_path)
    #         for img_name in class_files:
    #             img_path = os.path.join(class_folder_path, img_name)
    #             target = self.labels_map[class_folder]
    #             # 224 when using models pre-trained on ImageNet
    #             img = Image.open(img_path).resize((224, 224)).convert('RGB')
    #             img = np.array(img)
    #             self.data.append(img)
    #             self.targets.append(target)
    #     self.data = np.stack(self.data, axis=3)
    #     self.targets = np.array(self.targets)
    #     np.save(root+"/data.npy", self.data)
    #     np.save(root+"/targets.npy", self.targets)

    # Way 2: load the numpy files directly
    def __init__(self, root, sample_file, label_file, transform=None, target_transform=None) -> None:
        super(MyCoco, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(str(root) + '//' + sample_file)
        self.targets = np.load(str(root) + '//' + label_file).astype('long')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if len(self.data.shape) > 3:
            img, target = self.data[:, :, :, index], int(self.targets[index])
        else:
            img, target = self.data[:, :, index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MyTinyImageNet(data.Dataset):
    def __init__(self, root, sample_file, label_file, transform=None, target_transform=None) -> None:
        super(MyTinyImageNet, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(str(root) + '//' + sample_file)
        self.targets = np.load(str(root) + '//' + label_file).astype('long')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if len(self.data.shape) > 3:
            img, target = self.data[:, :, :, index], int(self.targets[index])
        else:
            img, target = self.data[:, :, index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def set_seed_torch(seed=0):
    # Set seed based on args.seed and the update number so that we get
    # reproducible results when resuming from checkpoints
    assert isinstance(seed, int)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, is_best, filename='checkpoint.pth'):    # forbidden to save as tar.gz
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.split(filename)[0] + '/checkpoint_best.pth')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
