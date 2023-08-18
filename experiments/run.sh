#!/bin/bash
cd /home/pengru/Contrastive_AutoEval

CUDA_VISIBLE_DEVICES=0 python run.py --data /home/pengru/data/Contrastive_AutoEval/datasets/ --train-dataset-name cifar10 \
--val-dataset-name cifar10 --arch densenet40-12 --batch-size 128 --epochs 300 --learning-rate 0.1 --seed 0 \
--out-dim 128 --temperature 0.07 --layers 40 --growth 12 --no-bottleneck --reduce 1.0 \
--num-classes 10 --save-dir /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3 \
--restore-file /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3/cifar10/checkpoint_0153.pth \
--optimizer SGD --momentum 0.9 --scheduler MultiStep --milestones 150,225 --gamma 0.1 \
--cl-model SimCLR --brightness 0.8 --contrast 0.8 --saturation 0.8 --hue 0.2 \
--ResizedCropScale 0.08,1.0 --data-setup cifar1 --claLoss-weight 1. --conLoss-weight 0.001