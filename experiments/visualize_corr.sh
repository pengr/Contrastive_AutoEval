#!/bin/bash
cd /home/pengru/Contrastive_AutoEval

python figure/visualize_corr.py --save-dir /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3/ \
--meta-dataset-name CIFAR-10 --epochs 150 --indice 6