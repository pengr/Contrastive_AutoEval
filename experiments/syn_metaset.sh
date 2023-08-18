#!/bin/bash
cd /home/pengru/Contrastive_AutoEval

python meta_set/synthesize_set_cifar.py --cifar-path /home/pengru/data/Contrastive_AutoEval/datasets/ \
--dataset-name cifar10 --metaset-size 500 --sampleset-size 10000 \
--metaset-dir /home/pengru/data/Contrastive_AutoEval/metasets/CIFAR10/metaset500_sampleset10k --workers 1