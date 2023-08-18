import numpy as np
from torchvision.transforms import transforms

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2, data_setup=None, train_trans=True):
        self.base_transform = base_transform
        self.n_views = n_views
        self.train_trans = train_trans

        self.train_transform = None
        self.test_transform = None

        # MNIST Setup
        if data_setup == "mnist1":
            self.normalize = transforms.Normalize((0.5,), (0.5,))
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

        # CIFAR Setup
        if data_setup == "cifar1":
            self.normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

        # COCO Setup
        if data_setup == "coco1":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize
            ])

        # TinyImageNet Setup
        if data_setup == "tinyimagenet":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                self.normalize
            ])

    def __call__(self, x):
        if self.train_trans:  # train set
            return [self.base_transform(x) for _ in range(self.n_views-1)] + [self.train_transform(x)]
        else:                 # test set
            return [self.base_transform(x) for _ in range(self.n_views - 1)] + [self.test_transform(x)]
