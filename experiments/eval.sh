cd /home/pengru/Contrastive_AutoEval

# 150epoch, cifar10.1
#CUDA_VISIBLE_DEVICES=0 python eval.py --data /home/pengru/data/Contrastive_AutoEval/datasets/ --test-dataset-name cifar10_1 --meta-dataset-name cifar10 \
#--metaset-dir /home/pengru/data/Contrastive_AutoEval/metasets/CIFAR10/metaset500_sampleset10k/dataset_default/ --metaset-numLim 2000 \
#--arch densenet40-12 --test-batch-size 44 --meta-batch-size 256 --seed 0 --out-dim 128 --temperature 0.07 \
#--layers 40 --growth 12 --no-bottleneck --reduce 1.0 \
#--num-classes 10 --save-dir /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3 \
#--restore-file /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3/cifar10/checkpoint_0150.pth \
#--cl-model SimCLR --brightness 0.8 --contrast 0.8 --saturation 0.8 --hue 0.2 --ResizedCropScale 0.08,1.0 --data-setup cifar1 \
#--claLoss-weight 1. --conLoss-weight 0.001 >> /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3/result.log

# 150epoch, cifar10-c
CUDA_VISIBLE_DEVICES=0 python eval_npy.py --data /home/pengru/data/Contrastive_AutoEval/datasets/ --test-dataset-name cifar10_c --meta-dataset-name cifar10 \
--metaset-dir /home/pengru/data/Contrastive_AutoEval/metasets/CIFAR10/metaset500_sampleset10k/dataset_default/ --metaset-numLim 2000 \
--arch densenet40-12 --test-batch-size 24 --meta-batch-size 256 --seed 0 --out-dim 128 --temperature 0.07 \
--layers 40 --growth 12 --no-bottleneck --reduce 1.0 \
--num-classes 10 --save-dir /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3 \
--restore-file /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3/cifar10/checkpoint_0150.pth \
--cl-model SimCLR --brightness 0.8 --contrast 0.8 --saturation 0.8 --hue 0.2 --ResizedCropScale 0.08,1.0 --data-setup cifar1 \
--claLoss-weight 1. --conLoss-weight 0.001 >> /home/pengru/data/Contrastive_AutoEval/checkpoints/CIFAR10/dqy_datasetup_conloss1e3/result.log