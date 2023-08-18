from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur, MocoGaussianBlur  # <fix>
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from utils import *


class ContrastiveLearningDataset:
    def __init__(self, args):
        self.root_folder = args.data
        self.args = args

    # @staticmethod
    def get_simclr_pipeline_transform(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(self.args.brightness * s, self.args.contrast * s,
                                              self.args.saturation * s, self.args.hue * s)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        data_transforms = None
        if self.args.cl_model == 'SimCLR':
            # the strength of RandomResizedCrop augmentation as r = (1-b) + (1-a)
            # scale:aug —> {(0.95, 1):0.05, (0.7, 1):0.3, (0.4, 1):0.6, (0.08, 1):0.92,
            # (0.2, 0.6):1.2, (0.2, 0.3):1.5,(0.02, 0.03):1.95}
            data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=size, scale=(eval(self.args.ResizedCropScale)[0], eval(self.args.ResizedCropScale)[1])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor()])

        elif self.args.cl_model == 'MoCo_V1':
            data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

        elif self.args.cl_model == 'MoCo_V2':
            data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([MocoGaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

        elif self.args.cl_model == 'BYOL':
            data_transforms = transforms.Compose([
                transforms.RandomApply([color_jitter], p=0.3),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
                transforms.RandomResizedCrop(size=size),
                transforms.ToTensor(),
                normalize
            ])
        return data_transforms

    def get_train_dataset(self, name, n_views, data_setup, train_trans=True):
        train_datasets = {### MNIST series
                          # load from the existing Meta-set MNIST, training sample size: 5w
                          'mnist': lambda: MyMNIST(self.root_folder + 'MNIST', 'train_data.npy', 'train_label.npy',
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(28),
                                                        n_views,
                                                        data_setup,
                                                        train_trans)),

                          # deprecated, torchvision MNIST, training sample size: 6w, RGB Image mode
                          'mnist_raw': lambda: MyMNISTRAW(self.root_folder, train=True,
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(28),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    download=True),  # Don't worry, it will skip if it already exists.

                          'fashion_mnist': lambda: MyFashionMNIST(self.root_folder, train=True,
                                                   transform=ContrastiveLearningViewGenerator(
                                                       self.get_simclr_pipeline_transform(28),
                                                       n_views,
                                                       data_setup,
                                                       train_trans),
                                                   download=True),

                          'k_mnist': lambda: MyKMNIST(self.root_folder, train=True,
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(28),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    download=True),
                          ### CIFAR series
                          # Only train test
                          'cifar10': lambda: datasets.CIFAR10(self.root_folder + 'CIFAR10', train=True,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(32),
                                                    n_views,
                                                    data_setup,
                                                    train_trans),
                                                download=True),    # Don't worry, it will skip if it already exists.

                          'cifar100': lambda: datasets.CIFAR100(self.root_folder + 'CIFAR100', train=True,
                                                                transform=ContrastiveLearningViewGenerator(
                                                                    self.get_simclr_pipeline_transform(32),
                                                                    n_views,
                                                                    data_setup,
                                                                    train_trans),
                                                                download=True),  # Don't worry, it will skip if it already exists.
                          # STL10 series
                          'stl10': lambda: datasets.STL10(self.root_folder, split='train',
                                                            transform=ContrastiveLearningViewGenerator(
                                                                self.get_simclr_pipeline_transform(96),
                                                                n_views,
                                                                data_setup,
                                                                train_trans),
                                                            download=True),

                          ### COCO series, cause here only 12 classes extracted from Orignal COCO Datasets,
                          # so we need to define a Mycoco Class by ourself, withou train=True
                          'coco': lambda: MyCoco(self.root_folder + "COCO/train2014/coco_cls_train_00000", 'data.npy', 'targets.npy',
                                                                    transform=ContrastiveLearningViewGenerator(
                                                                        self.get_simclr_pipeline_transform(224),
                                                                        n_views,
                                                                        data_setup,
                                                                        train_trans)),

                            ### TinyImageNet series, use torchvision.dataset.ImageFolder to load this datasets
                            'tinyimagenet': lambda: datasets.ImageFolder(self.root_folder + "tiny-imagenet-200/train",
                                                transform=ContrastiveLearningViewGenerator(
                                                       self.get_simclr_pipeline_transform(64),
                                                       n_views,
                                                       data_setup,
                                                       train_trans)),
        }
        try:
            dataset_fn = train_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

    # validation dataset for evaluating the two model acc of each epoch
    def get_val_dataset(self, name, n_views, data_setup, train_trans=False):
        val_datasets = {### MNIST series
                        'mnist': lambda: MyMNIST(self.root_folder + 'MNIST', 'test_data.npy', 'test_label.npy',
                                     transform=ContrastiveLearningViewGenerator(
                                         self.get_simclr_pipeline_transform(28),
                                         n_views,
                                         data_setup,
                                         train_trans)
                                     ),

                        # <fix>, deprecated, torchvision MNIST, training sample size: 6w, RGB Image mode
                        'mnist_raw': lambda: MyMNISTRAW(self.root_folder, train=False,
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(28),
                                                            n_views,
                                                            data_setup,
                                                            train_trans),
                                                        download=True),

                        'fashion_mnist': lambda: MyFashionMNIST(self.root_folder, train=False,
                                                                transform=ContrastiveLearningViewGenerator(
                                                                    self.get_simclr_pipeline_transform(28),
                                                                    n_views,
                                                                    data_setup,
                                                                    train_trans),
                                                                download=True),

                        'k_mnist': lambda: MyKMNIST(self.root_folder, train=False,
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(28),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    download=True),

                        'cifar10': lambda: datasets.CIFAR10(self.root_folder + 'CIFAR10', train=False,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(32),
                                                    n_views,
                                                    data_setup,
                                                    train_trans),
                                                download=True),

                        'cifar100': lambda: MyCIFAR100(self.root_folder + 'CIFAR100', 'test_data.npy', 'test_label.npy',
                                                     transform=ContrastiveLearningViewGenerator(
                                                         self.get_simclr_pipeline_transform(32),
                                                         n_views,
                                                         data_setup,
                                                         train_trans),
                                                     ),

                        # COCO series, cause here only 12 classes extracted from Orignal COCO Datasets,
                        # so we need to define a Mycoco Class by ourself
                        'coco': lambda: MyCoco(self.root_folder + "COCO/val2014/coco_cls_train_00000", 'data.npy', 'targets.npy',
                                               transform=ContrastiveLearningViewGenerator(
                                                   self.get_simclr_pipeline_transform(224),
                                                   n_views,
                                                   data_setup,
                                                   train_trans),
                                               ),

                        ### TinyImageNet series, use torchvision.dataset.ImageFolder to load this datasets
                        'tinyimagenet': lambda: datasets.ImageFolder(self.root_folder + "tiny-imagenet-200/val",
                                                                    transform=ContrastiveLearningViewGenerator(
                                                                        self.get_simclr_pipeline_transform(64),
                                                                        n_views,
                                                                        data_setup,
                                                                        train_trans)),

                        'stl10': lambda: datasets.STL10(self.root_folder, split='test',  #  test mode
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(96),
                                                            n_views,
                                                            data_setup,
                                                            train_trans),
                                                        download=True)
                        }

        try:
            dataset_fn = val_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

    # seed sets for synthesizing meta dataset, usually the original dataset itself
    def get_seed_dataset(self, root_folder, name):
        NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        te_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(*NORM)])

        seed_datasets = {## MNIST series, Single Channels
                        'mnist': lambda: MNIST_bg(root_folder + 'MNIST', 'test_data.npy', 'test_label.npy'),

                        'fashion_mnist': lambda: datasets.FashionMNIST(root_folder, train=False, download=True),

                        'k_mnist': lambda: datasets.KMNIST(root_folder, train=False, download=True),

                        ## CIFAR series
                        'cifar10': lambda: datasets.CIFAR10(root_folder + 'CIFAR10', train=False, download=True, transform=None),

                        'cifar100': lambda: MyCIFAR100(root_folder + 'CIFAR100', 'test_data.npy', 'test_label.npy',
                                                       transform=te_transforms),

                        'slt10': lambda: datasets.STL10(root_folder, split='test', download=True, transform=te_transforms),

                        ## COCO series, use its source code

                        ## TinyImageNet series, use torchvision.dataset.ImageFolder to load this datasets
                        'tinyimagenet': lambda: datasets.ImageFolder(root_folder + "tiny-imagenet-200/val")
        }

        try:
            dataset_fn = seed_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

    # meta sets for training a regression model (If just need a correlation value, don't need this step)
    def get_meta_dataset(self, root_folder, name, n_views, data_setup, train_trans=False):
        meta_datasets = {### MNIST series, Single Channels
                        'mnist': lambda: MyMNIST(root_folder, 'test_data.npy', 'test_label.npy',
                                                 transform=ContrastiveLearningViewGenerator(
                                                     self.get_simclr_pipeline_transform(28),
                                                     n_views,
                                                     data_setup,
                                                     train_trans)
                                                 ),

                        ### CIFAR series
                        'cifar10': lambda: MyCIFAR10(root_folder, 'test_data.npy', 'test_label.npy',
                                                       transform=ContrastiveLearningViewGenerator(
                                                           self.get_simclr_pipeline_transform(32),
                                                           n_views,
                                                           data_setup,
                                                           train_trans),
                                                       ),

                        'cifar100': lambda: MyCIFAR100(root_folder, 'test_data.npy', 'test_label.npy',
                                                       transform=ContrastiveLearningViewGenerator(
                                                           self.get_simclr_pipeline_transform(32),
                                                           n_views,
                                                           data_setup,
                                                           train_trans),
                                                       ),

                        ### COCO series, use its source code
                        'coco': lambda: MyCoco(root_folder, 'data.npy', 'targets.npy',
                                               transform=ContrastiveLearningViewGenerator(
                                                   self.get_simclr_pipeline_transform(224),
                                                   n_views,
                                                   data_setup,
                                                   train_trans),
                                               ),

                        'tinyimagenet': lambda: MyTinyImageNet(root_folder, 'test_data.npy', 'test_label.npy',
                                                   transform=ContrastiveLearningViewGenerator(
                                                       self.get_simclr_pipeline_transform(64),
                                                       n_views,
                                                       data_setup,
                                                       train_trans),
                                                   ),
        }

        try:
            dataset_fn = meta_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

    # actually it's unseen target test dataset for evaluating model performance predictions
    def get_test_dataset(self, root_folder, name, n_views, data_setup, train_trans=False):
        test_datasets = {'svhn': lambda: datasets.SVHN(root_folder + 'SVHN', split='test',
                                                       transform=transforms.Compose([
                                                           transforms.Resize(28),
                                                           # convert into the MNIST-type size
                                                           ContrastiveLearningViewGenerator(
                                                               self.get_simclr_pipeline_transform(28),
                                                               n_views,
                                                               data_setup,
                                                               train_trans)]),
                                                       download=True),

                         'usps': lambda: datasets.USPS(root_folder + 'USPS', train=False,
                                                       transform=transforms.Compose([
                                                           transforms.Grayscale(num_output_channels=3),
                                                           # convert into the MNIST-type channels
                                                           transforms.Resize(28),
                                                           # convert into the MNIST-type size
                                                           ContrastiveLearningViewGenerator(
                                                               self.get_simclr_pipeline_transform(28),
                                                               n_views,
                                                               data_setup,
                                                               train_trans)]),
                                                       download=True),

                         'cifar10_1': lambda: CIFAR10_1(root_folder + 'CIFAR10_1',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(32),
                                                            n_views,
                                                            data_setup,
                                                            train_trans),
                                                        ),

                         'cifar100': lambda: MyCIFAR100(root_folder + 'CIFAR100', 'unseen_test_data.npy',
                                                        'unseen_test_label.npy',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(32),
                                                            n_views,
                                                            data_setup,
                                                            train_trans),
                                                        ),

                         'caltech': lambda: MyCoco(root_folder + 'Caltech256', 'data.npy', 'targets.npy',
                                                   transform=ContrastiveLearningViewGenerator(
                                                       self.get_simclr_pipeline_transform(224),
                                                       n_views,
                                                       data_setup,
                                                       train_trans),
                                                   ),

                         'caltech_A': lambda: MyCoco(root_folder + 'c_all_transformA','data.npy', 'targets.npy',
                                                   transform=ContrastiveLearningViewGenerator(
                                                       self.get_simclr_pipeline_transform(224),
                                                       n_views,
                                                       data_setup,
                                                       train_trans),
                                                   ),

                         'caltech_B': lambda: MyCoco(root_folder + 'c_all_transformB','data.npy', 'targets.npy',
                                                   transform=ContrastiveLearningViewGenerator(
                                                       self.get_simclr_pipeline_transform(224),
                                                       n_views,
                                                       data_setup,
                                                       train_trans),
                                                   ),

                         'caltech_C': lambda: MyCoco(root_folder + 'c_all_transformC','data.npy', 'targets.npy',
                                                     transform=ContrastiveLearningViewGenerator(
                                                         self.get_simclr_pipeline_transform(224),
                                                         n_views,
                                                         data_setup,
                                                         train_trans),
                                                     ),

                         'pascal': lambda: MyCoco(root_folder + 'PASCAL_test', 'data.npy', 'targets.npy',
                                                  transform=ContrastiveLearningViewGenerator(
                                                      self.get_simclr_pipeline_transform(224),
                                                      n_views,
                                                      data_setup,
                                                      train_trans),
                                                  ),

                         'pascal_A': lambda: MyCoco(root_folder + 'p_all_transformA','data.npy', 'targets.npy',
                                                  transform=ContrastiveLearningViewGenerator(
                                                      self.get_simclr_pipeline_transform(224),
                                                      n_views,
                                                      data_setup,
                                                      train_trans),
                                                  ),

                         'pascal_B': lambda: MyCoco(root_folder + 'p_all_transformB','data.npy', 'targets.npy',
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(224),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    ),

                         'pascal_C': lambda: MyCoco(root_folder + 'p_all_transformC','data.npy', 'targets.npy',
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(224),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    ),

                         'imagenet': lambda: MyCoco(root_folder + 'ImageNet', 'data.npy', 'targets.npy',
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(224),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    ),

                         'imagenet_A': lambda: MyCoco(root_folder + 'i_all_transformA', 'data.npy', 'targets.npy',
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(224),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    ),

                         'imagenet_B': lambda: MyCoco(root_folder + 'i_all_transformB', 'data.npy', 'targets.npy',
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(224),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    ),

                         'imagenet_C': lambda: MyCoco(root_folder + 'i_all_transformC', 'data.npy', 'targets.npy',
                                                      transform=ContrastiveLearningViewGenerator(
                                                          self.get_simclr_pipeline_transform(224),
                                                          n_views,
                                                          data_setup,
                                                          train_trans),
                                                      ),

                         ### TinyImageNet series, use torchvision.dataset.ImageFolder to load this datasets
                         'tinyimagenet_c': lambda: datasets.ImageFolder(root_folder,
                                                                        transform=ContrastiveLearningViewGenerator(
                                                                            self.get_simclr_pipeline_transform(64),
                                                                            n_views,
                                                                            data_setup,
                                                                            train_trans)),

                         'cifar10_c': lambda: CIFAR10_c(root_folder,
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(32),
                                                            n_views,
                                                            data_setup,
                                                            train_trans),
                                                        ),

                         'cifar100_c': lambda: CIFAR100_c(root_folder,
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(32),
                                                            n_views,
                                                            data_setup,
                                                            train_trans),
                                                        ),
                         }

        try:
            dataset_fn = test_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()