import os
import random
import sys
sys.path.append(".")
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
import argparse
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset

parser = argparse.ArgumentParser(description='Synthesize TinyImageNet Meta-set')
parser.add_argument('--tinyimagenet-path', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/datasets/',
                    help='path to tinyimagenet dataset')
parser.add_argument('--dataset-name', default='tinyimagenet', help='seed dataset name', choices=['tinyimagenet'])
parser.add_argument('--metaset-size', default=700, type=int, metavar='N', help='the number of sample sets')
parser.add_argument('--sampleset-size', default=400, type=int, metavar='N', help='the size of each sample set')
parser.add_argument('--metaset-dir', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/metasets/TinyImageNet',
                    help='path to save the generated meta-set')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')

# ===================================================== #
# -----------     load original dataset     ----------- #
# ===================================================== #
'''An example to load original datasets (based on Pytorch's dataloader)'''
args = parser.parse_args()
args.data = args.tinyimagenet_path
dataset = ContrastiveLearningDataset(args)  # use the cl-model type to choose the transformation type
teset = dataset.get_seed_dataset(args.tinyimagenet_path, args.dataset_name)  # <fix>
teset_raw = teset.samples
teset_label_raw = teset.targets
print('Loaded original set')

# ===================================================== #
# -----------     Image Transformations     ----------- #
# ===================================================== #
'''
In Frechet Distance paper, the transformations are: 
{Autocontrast, Brightness, Color, ColorSolarize, Contrast, Rotation, Sharpness, TranslateX/Y}
The users can customize the transformation list based on the their own data.
The users can use more transformations for the selection.
We refer the readers to https://imgaug.readthedocs.io/ for more details of transformations.
Here, we provide 3 examples, hope you enjoy it!
'''
# Default
# {Autocontrast, Brightness, Color, ColorSolarize, Contrast, Rotation, Sharpness, TranslateX/Y}
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
list = [
    iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
    iaa.Multiply((0.1, 1.9), per_channel=0.2),  # make some images brighter and some darker
    iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
    iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
    iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-30, 30)
    )), # add affine transformation
    iaa.Sharpen(alpha=(0.1, 1.0)),  # apply a sharpening filter kernel to images
]

# GroupA
# list = [
#     iaa.Grayscale(alpha=(0.0, 0.5)),  # remove colors with varying strengths
#     iaa.ElasticTransformation(alpha=(0.5, 2.5), sigma=(0.25, 0.5)),  # move pixels locally around with random strengths
#     iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # distort local areas with varying strength
#     iaa.Invert(0.05, per_channel=True),  # invert color channels
#     iaa.pillike.FilterBlur(),  # apply a blur filter kernel to images
#     iaa.pillike.EnhanceBrightness(),  # change the brightness of images
#     iaa.Fog(),  # add fog to images
#     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025 * 255), per_channel=0.5)  # Add gaussian noise to some images
# ]

# GroupB
# list = [
#     iaa.LinearContrast((0.5, 1.5), per_channel=0.5),  # improve or worsen the contrast of images
#     iaa.Rain(speed=(0.1, 0.5)),  # add rain to small images
#     iaa.JpegCompression(compression=(70, 99)), # degrade the quality of images by JPEG-compressing them
#     iaa.GaussianBlur(sigma=(0.0, 3.0)),  # augmenter to blur images using gaussian kernels
#     iaa.pillike.EnhanceSharpness(),  # alpha-blend two image sources using an alpha/opacity value
#     iaa.MultiplyHue((0.5, 1.5)),  # change the sharpness of images
#     iaa.Emboss(alpha=(0.0, 0.5), strength=(0.5, 1.5)),  # emboss an image, then overlay the results with the original
#     iaa.AddToSaturation((-50, 50))  # add random values to the saturation of images
# ]

# GroupC
# add more transformations into the list based on the users' need
# sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# list = [
#     iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
#     iaa.Multiply((0.1, 1.9), per_channel=0.2),  # make some images brighter and some darker
#     iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
#     iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
#     iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast of images
#     sometimes(iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#         rotate=(-30, 30)
#     )), # add affine transformation
#     iaa.Sharpen(alpha=(0.0, 1.0)),  # apply a sharpening filter kernel to images
#     iaa.LinearContrast((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast of images
#     iaa.Rain(speed=(0.1, 0.3)),  # add rain to small images
#     iaa.JpegCompression(compression=(70, 99)), # degrade the quality of images by JPEG-compressing them
#     iaa.pillike.FilterDetail(),  # apply a detail enhancement filter kernel to images
#     iaa.pillike.EnhanceSharpness(),  # alpha-blend two image sources using an alpha/opacity value
#     iaa.MultiplyHue((0.8, 1.2)),  # change the sharpness of images
#     iaa.Emboss(alpha=(0.0, 0.5), strength=(0.5, 1.0)),  # emboss an image, then overlay the results with the original
#     iaa.AddToSaturation((-25, 25))  # add random values to the saturation of images
# ]

'''
In Rotation paper, the transformations are: 
{Sharpness, translateX/Y, Color, Autocontrast, Brightness, Rotation}.
The users can customize the transformation list based on the their own data.
The users can use more transformations for the selection.
We refer the readers to https://imgaug.readthedocs.io/ for more details of transformations.
'''
# {Sharpness, translateX/Y, Color, Autocontrast, Brightness, Rotation}
# GroupD
# list = [
#     iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
#     iaa.Sharpen(alpha=(0.0, 1.0)),  # apply a sharpening filter kernel to images
#     iaa.ChangeColorTemperature((1100, 10000 // 2)),  # change the temperature to a provided Kelvin value.
#     iaa.pillike.Equalize(),  # equalize the image histogram
#     iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
#     iaa.Multiply((0.8, 1.2), per_channel=0.2),  # make some images brighter and some darker
#     iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
#     iaa.Grayscale(alpha=(0.0, 0.5)),  # remove colors with varying strength
# ]

# ===================================================== #
# -----------  Generate Synthetic datasets  ----------- #
# ===================================================== #
'''
Generate 800 and 200 synthetic datasets for training and validation, respectively
'''
tesize = args.sampleset_size
num_sets = args.metaset_size
# Change to your path
try:
    os.makedirs(f'{args.metaset_dir}/dataset_default')
except:
    print('Alread has this path')


def makeTinyimagenet(num, list, teset_raw, teset_label_raw, args):
    num_sel = 3  # use more transformation to introduce dataset diversity
    list_sel = random.sample(list, int(num_sel))
    random.shuffle(list_sel)
    seq = iaa.Sequential(list_sel)

    # Directly for i in datasets.ImageFolder(tinyimagenet), no difference from manually doing Pil.Image.open
    # new_data = np.zeros(teset_raw.shape)
    new_data = []
    new_target = []
    for i in range(tesize):
        data = np.array(Image.open(teset_raw[i][0]).convert('RGB'))  # tinyimagenet samples format [path, target], convert jpg image to return a PIL Image
        ia.seed(i + num * tesize)  # add random for each dataset
        new_data.append(seq(image=data))
        new_target.append(teset_label_raw[i])
    new_data = np.stack(new_data, axis=3)
    new_target = np.array(new_target)
    # firstly create the directory, then save npy files
    try:
        os.makedirs(f'{args.metaset_dir}/dataset_default/new_{str(num).zfill(3)}')
    except:
        None
    np.save(f'{args.metaset_dir}/dataset_default/new_{str(num).zfill(3)}/test_data.npy', new_data)
    np.save(f'{args.metaset_dir}/dataset_default/new_{str(num).zfill(3)}/test_label.npy', new_target)


pool = None
pbar = tqdm(total=len(range(args.metaset_size)))
update = lambda *args: pbar.update()

# two ways of synthesising coco metasets without background replacing
# ----------- way 1: store into numpy format ----------- #
if args.workers > 1:
    pool = Pool(processes=args.workers)
    for num in range(num_sets):
        pool.apply_async(
            makeTinyimagenet,
            args=(
                num, list, teset_raw, teset_label_raw, args
            ),
            callback=update
        )
    pool.close()
    pool.join()
else:
    for num in range(num_sets):
        makeTinyimagenet(num, list, teset_raw, teset_label_raw, args)
print('Finished, thanks!')