import os
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Synthesize COCO Meta-set')
parser.add_argument('--coco-path', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/datasets/extra_data',
                    help='path to coco dataset')
parser.add_argument('--metaset-size', default=200, type=int, metavar='N', help='the number of sample sets')
parser.add_argument('--sampleset-size', default=600, type=int, metavar='N', help='the size of each sample set')
parser.add_argument('--metaset-dir', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/metasets/COCO/coco_cls_val_b_',
                    help='path to save the generated meta-set')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')

# ===================================================== #
# -----------     Image Transformations     ----------- #
# ===================================================== #

'''
In our paper, the transformations are:
{Autocontrast, Brightness, Color, ColorSolarize, Contrast, Rotation, Sharpness, TranslateX/Y}.
Other transformations can be also used.
The users can customize the transformation list based on the their own data.
We refer the readers to https://imgaug.readthedocs.io/ for more details of transformations.
Here, we provide 3 examples, hope you enjoy it!
'''
# Default
# {Autocontrast, Brightness, Color, ColorSolarize, Contrast, Rotation, Sharpness, TranslateX/Y}
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
list = [
    iaa.pillike.Autocontrast(), # adjust contrast by cutting off p% of lowest/highest histogram values
    iaa.Multiply((0.1, 1.9), per_channel=0.2), # make some images brighter and some darker
    iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
    iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
    iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-30, 30)
    )),  # add affine transformation
    iaa.Sharpen(alpha=(0.1, 1.0)), # apply a sharpening filter kernel to images
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

# add more transformations into the list based on the users' need
# sometimes = lambda aug: iaa.Sometimes(0.5, aug)
list1 = [
    iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
    iaa.Multiply((0.1, 1.9), per_channel=0.2),  # make some images brighter and some darker
    iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
    iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
    iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast of images
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-30, 30)
    )), # add affine transformation
    iaa.Sharpen(alpha=(0.0, 1.0)),  # apply a sharpening filter kernel to images
    iaa.LinearContrast((0.8, 1.2), per_channel=0.5),  # improve or worsen the contrast of images
    iaa.Rain(speed=(0.1, 0.3)),  # add rain to small images
    iaa.JpegCompression(compression=(70, 99)), # degrade the quality of images by JPEG-compressing them
    iaa.pillike.FilterDetail(),  # apply a detail enhancement filter kernel to images
    iaa.pillike.EnhanceSharpness(),  # alpha-blend two image sources using an alpha/opacity value
    iaa.MultiplyHue((0.8, 1.2)),  # change the sharpness of images
    iaa.Emboss(alpha=(0.0, 0.5), strength=(0.5, 1.0)),  # emboss an image, then overlay the results with the original
    iaa.AddToSaturation((-25, 25))  # add random values to the saturation of images
]

# ===================================================== #
# -----------       Load COCO Dataset       ----------- #
# ===================================================== #
args = parser.parse_args()
dataDir = args.coco_path  # COCO dataset path, COCO path: PROJECT_DIR/extra_data/
dataType = 'val2014'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

# ===================================================== #
# -----------      Generate Sample Sets     ----------- #
# ===================================================== #

# 12 classes; shared across ImageNet-Pascal-COCO-Caltech
target_list = ['airplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'dog', 'horse', 'motorcycle', 'person', 'tv']


# select bbox following the practice in http://ai.bu.edu/visda-2017/
thresh = 120  # for bbox's W and H
crop_pixels = 50
b_thresh = 120  # crop background
num_sets = args.metaset_size  # generate 800 and 200 sample sets for training and validation

def makeCocoRotation_npy(indice_set, list, args, target_list, coco):
    if indice_set <= num_sets * 3/20:
        num_sel = 3  # select more transformations to make sample set diverse
        list_sel = random.sample(list, int(num_sel))
        random.shuffle(list_sel)
        seq = iaa.Sequential(list_sel)
    else:
        random.shuffle(list1)
        seq = iaa.Sequential(list1)
    save_rb_path = args.metaset_dir + 'coco_cls_val_b_' + str(indice_set).zfill(5)  # <fix>, the users need to change the path
    if not os.path.exists(save_rb_path):
        os.makedirs(save_rb_path)

    data = []
    targets = []
    # generate images category by category
    # target is the current selected category
    for cls_indice, target in enumerate(target_list):
        im_seq = 1  # how many images are selected for the current category
        ss_Id = coco.getCatIds(catNms=[target])[0]  # get the category_id of the current category
        imgIds = coco.getImgIds(catIds=ss_Id)  # find the images that contains the current category
        random.shuffle(imgIds)

        # handle image that contains the current category
        for img_id in imgIds:
            if im_seq <= args.sampleset_size:  # 600->100, select at most 600 images for each category
                imgIds = coco.getImgIds(imgIds=img_id)
                img = coco.loadImgs(imgIds)[0]
                I = cv2.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
                hight = I.shape[0]
                width = I.shape[1]
                # load bbox and segmentation annotations for the current image
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=ss_Id, iscrowd=False)
                for each_ann_id in annIds:
                    anns = coco.loadAnns(each_ann_id)
                    if (len(anns) != 0):
                        if im_seq <= args.sampleset_size:  # 600->100, select at most 600 images for each category
                            for ann in anns:
                                if ann['category_id'] == ss_Id:  # choose the object that is from the current category
                                    # crop object
                                    x, y, w, h = ann['bbox']
                                    if w > thresh and h > thresh:
                                        x1 = max(int(float(x)) - crop_pixels, 1)
                                        x2 = min(int(float(x)) + int(float(w)) + crop_pixels, width - 1)
                                        y1 = max(int(float(y)) - crop_pixels, 1)
                                        y2 = min(int(float(y)) + int(float(h)) + crop_pixels, hight - 1)
                                        I_cp = I[y1: y2, x1: x2]
                                        try:
                                            # # apply image transformation to introduce more visual changes
                                            # # control the randomness for each dataset
                                            ia.seed(im_seq + cls_indice * 600 + indice_set * 8000)
                                            I_cp_nb = seq(image=I_cp)
                                            img = Image.fromarray(I_cp_nb).resize((224, 224)).convert('RGB')
                                            img = np.array(img)

                                            # save image
                                            data.append(img)
                                            targets.append(cls_indice)
                                            # how many images are selected for the current category
                                            im_seq = im_seq + 1
                                        except:
                                            print('jump an object')
    data = np.stack(data, axis=3)
    targets = np.array(targets)
    np.save(save_rb_path + "/data.npy", data)
    np.save(save_rb_path + "/targets.npy", targets)


def makeCocoRotation_jpg(indice_set, list, args, target_list, coco):
    if indice_set <= num_sets * 3/20:
        num_sel = 3  # select more transformations to make sample set diverse
        list_sel = random.sample(list, int(num_sel))
        random.shuffle(list_sel)
        seq = iaa.Sequential(list_sel)
    else:
        random.shuffle(list1)
        seq = iaa.Sequential(list1)
    save_rb_path = args.metaset_dir + 'coco_cls_val_b_' + str(indice_set).zfill(5)  # the users need to change the path
    if not os.path.exists(save_rb_path):
        os.makedirs(save_rb_path)

    # generate images category by category
    # target is the current selected category
    for cls_indice, target in enumerate(target_list):
        target_rb_dir = save_rb_path + '/' + target
        if not os.path.exists(target_rb_dir):
            os.makedirs(target_rb_dir)

        im_seq = 1  # how many images are selected for the current category
        ss_Id = coco.getCatIds(catNms=[target])[0]  # get the category_id of the current category
        imgIds = coco.getImgIds(catIds=ss_Id)  # find the images that contains the current category
        random.shuffle(imgIds)

        # handle image that contains the current category
        for img_id in imgIds:
            if im_seq <= args.sampleset_size:  # 600->100, select at most 600 images for each category
                imgIds = coco.getImgIds(imgIds=img_id)
                img = coco.loadImgs(imgIds)[0]
                I = cv2.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
                hight = I.shape[0]
                width = I.shape[1]
                # load bbox and segmentation annotations for the current image
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=ss_Id, iscrowd=False)
                for each_ann_id in annIds:
                    anns = coco.loadAnns(each_ann_id)
                    if (len(anns) != 0):
                        if im_seq <= args.sampleset_size:  # 600->100, select at most 600 images for each category
                            for ann in anns:
                                if ann['category_id'] == ss_Id:  # choose the object that is from the current category
                                    # crop object
                                    x, y, w, h = ann['bbox']
                                    if w > thresh and h > thresh:
                                        x1 = max(int(float(x)) - crop_pixels, 1)
                                        x2 = min(int(float(x)) + int(float(w)) + crop_pixels, width - 1)
                                        y1 = max(int(float(y)) - crop_pixels, 1)
                                        y2 = min(int(float(y)) + int(float(h)) + crop_pixels, hight - 1)
                                        I_cp = I[y1: y2, x1: x2]
                                        try:
                                            # # apply image transformation to introduce more visual changes
                                            # # control the randomness for each dataset
                                            ia.seed(im_seq + cls_indice * 600 + indice_set * 8000)
                                            I_cp_nb = seq(image=I_cp)

                                            # save image
                                            cv2.imwrite(target_rb_dir + '/' + '{:09d}.jpg'.format(im_seq), I_cp_nb)
                                            # how many images are selected for the current category
                                            im_seq = im_seq + 1
                                        except:
                                            print('jump an object')


pool = None
pbar = tqdm(total=len(range(args.metaset_size)))
update = lambda *args: pbar.update()

# two ways of synthesising coco metasets without background replacing
# ----------- way 1: store into numpy format ----------- #
if args.workers > 1:
    pool = Pool(processes=args.workers)
    for indice_set in range(num_sets):
        pool.apply_async(
            makeCocoRotation_npy,
            args=(
                indice_set, list, args, target_list, coco
            ),
            callback=update
        )
    pool.close()
    pool.join()
else:
    for indice_set in range(num_sets):
        makeCocoRotation_npy(indice_set, list, args, target_list, coco)

# ----------- way 2: store into jpg format ----------- #
# if args.workers > 1:
#     pool = Pool(processes=args.workers)
#     for indice_set in range(num_sets):
#         pool.apply_async(
#             makeCocoRotation_jpg,
#             args=(
#                 indice_set, list, args, target_list, coco
#             ),
#             callback=update
#         )
#     pool.close()
#     pool.join()
# else:
#     for indice_set in range(num_sets):
#         makeCocoRotation_jpg(indice_set, list, args, target_list, coco)