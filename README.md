# CAME: Contrastive Automated Model Evaluation 
## [[Paper]](https://arxiv.org/abs/2308.11111)
![](https://github.com/pengr/Contrastive_AutoEval/blob/master/Our_Model.png)


## PyTorch Implementation

This repository contains:

- the PyTorch implementation of CAME
- the example on CIFAR-10 setup
- Contrastive Accuracy calculation and linear regression methods
- MNIST, CIFAR-100, COCO and TinyImageNet Setups (use [imgaug](https://imgaug.readthedocs.io/en/latest/) to generate Meta-set).
  Please see ```PROJECT_DIR/meta_set/```

Please follow the instruction below to install it and run the experiment demo.

### Prerequisites
* Linux (tested on Ubuntu 16.04LTS)
* NVIDIA GPU + CUDA CuDNN (tested on Tesla V100)
* [SimCLR-v1](https://github.com/sthalles/SimCLR) our codebase
* [MNIST Dataset](https://drive.google.com/file/d/1wq8pIdayAbCu5MBfT1M38BATcShsaaeq/view?usp=sharing) (download and unzip to ```PROJECT_DIR/datasets/MNIST```)
* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (download and unzip to ```PROJECT_DIR/datasets/CIFAR10```)
* [CIFAR10.1 Dataset](https://github.com/modestyachts/CIFAR-10.1) (download and unzip to ```PROJECT_DIR/datasets/CIFAR10_1```)
* [CIFAR-10-C Dataset](https://zenodo.org/record/2535967#.Y-3ggHZBx3g) (download and unzip to ```PROJECT_DIR/datasets/CIFAR-10-C```)
* [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (download and unzip to ```PROJECT_DIR/datasets/CIFAR100```)
* [CIFAR-100-C Dataset](https://zenodo.org/record/3555552#.Y-3gwHZBx3g) (download and unzip to ```PROJECT_DIR/datasets/CIFAR-100-C```)
* [COCO 2017 Dataset](http://cocodataset.org) (download and unzip to ```PROJECT_DIR/extra_data```)
* [PASCAL Dataset](http://host.robots.ox.ac.uk/pascal/VOC/) (download and unzip to ```PROJECT_DIR/datasets/PASCAL```)
* [Caltech256 Dataset](https://data.caltech.edu/records/nyy15-4j048) (download and unzip to ```PROJECT_DIR/datasets/Caltech256```)
* [ImageNet Dataset](https://image-net.org/challenges/LSVRC/2013/2013-downloads.php) (download and unzip to ```PROJECT_DIR/datasets/ImageNet```)
* [TinyImageNet Dataset](http://cs231n.stanford.edu/tiny-imagenet-200.zip) (download and unzip to ```PROJECT_DIR/datasets/tiny-imagenet-200```)
* [TinyImageNet-C Dataset](https://zenodo.org/record/2469796#.Y-3gynZBx3g) (download and unzip to ```PROJECT_DIR/datasets/Tiny-ImageNet-C```)
* SHVN,USPS,FashionMNIST,KMNIST,STL10 are the ready-made datasets in torchvision.datasets package 
* All -C Operation can refers to [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://github.com/hendrycks/robustness)
* You might need to change the file paths, and please be sure you change the corresponding paths in the codes as well  
* Please see more details about COCO setup in https://github.com/Simon4Yan/Meta-set/issues/2


## Getting started
0. Install dependencies 
    ```bash
    # COCOAPI
    cd $DIR/libs
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py build_ext install
    # CAME
    conda env create --name contrastive_autoeval --file environment.yml
    conda activate contrastive_autoeval
    ```

1. Co-train classifier
    ```bash
    # Save as "PROJECT_DIR/checkpoints/CIFAR10/checkpoint.pth"
    python run.py
    ```
    
2. Creat Meta-set
    ```bash
    # By default it creates 400 sample sets
    python meta_set/synthesize_set_cifar.py
    ```
   
3. Test classifier on Meta-set and save the fitted regression model
    ```bash
    # Get "PROJECT_DIR/checkpoints/CIFAR10/accuracy_cifar.npy" file
    python regression.py
    ```

4. Eval on unseen test sets by regression model
    ```bash
    # 1) You will see Rank correlation and Pearsons correlation
    # 2) The absolute error of linear regression is also shown
    python eval.py
    ``` 

5. Correlation study
    ```bash
    # You will see correlation.pdf;
    python figure/visualize_corr.py
        
## Citation
If you use the code in your research, please cite:
```bibtex
    @inproceedings{peng2023contrastive,
    author={Ru Peng, Qiuyang Duan, Haobo Wang, Jiachen Ma, Yanbo Jiang, Yongjun Tu, Xiu Jiang, Junbo Zhao},
    title     = {CAME: Contrastive Automated Model Evaluation},
    booktitle = {Proc. ICCV},
    year      = {2023},
    }
```

## License
MIT
