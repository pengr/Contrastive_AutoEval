import os
import pathlib
import sys
sys.path.append(".")
import PIL
import numpy as np
from tqdm import trange
from meta_set.util import random_rotation_new
from imageio import imsave
import argparse
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset

parser = argparse.ArgumentParser(description='Synthesize MNIST Meta-set')
parser.add_argument('--mnist-path', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/datasets/',
                    help='path to mnist dataset')
parser.add_argument('--coco-path', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/datasets/extra_data/train2014/',
                    help='path to coco dataset')
parser.add_argument('--dataset-name', default='mnist', help='seed dataset name', choices=['mnist', 'fashion_mnist', 'k_mnist'])
parser.add_argument('--metaset-size', default=200, type=int, metavar='N', help='the number of sample sets')
parser.add_argument('--sampleset-size', default=10000, type=int, metavar='N', help='the image number in each sample set')
parser.add_argument('--metaset-dir', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/metasets/MNIST',
                    help='path to save the generated meta-set')


def creat_bg(input_image, img, change_colors=False):
    # Rotate image
    input_image = random_rotation_new(input_image)
    # Extend to RGB
    input_image = np.expand_dims(input_image, 2)
    input_image = input_image / 255.0
    input_image = np.concatenate([input_image, input_image, input_image], axis=2)

    # Convert the MNIST images to binary
    img_binary = (input_image > 0.5)

    # Take a random crop of the Lena image (background)
    x_c = np.random.randint(0, img.size[0] - 28)
    y_c = np.random.randint(0, img.size[1] - 28)
    image_bg = img.crop((x_c, y_c, x_c + 28, y_c + 28))
    # Conver the image to float between 0 and 1
    image_bg = np.asarray(image_bg) / 255.0

    if change_colors:
        # Change color distribution; this leads to more diverse changes than transformations
        for j in range(3):
            image_bg[:, :, j] = (image_bg[:, :, j] + np.random.uniform(0, 1)) / 2.0

    # Invert the colors at the location of the number
    image_bg[img_binary] = 1 - image_bg[img_binary]

    image_bg = image_bg / float(np.max(image_bg)) * 255
    return image_bg


def makeMnistbg(metaset_dir, img, num=1):  # <fix>
    """
    Change the background of  MNIST images
    Select all testing samples from MNIST
    Store in numpy file for fast reading
    """

    index = str(num).zfill(5)
    np.random.seed(0)

    # Empty arrays
    test_data = np.zeros([28, 28, 3, 10000])
    test_label = np.zeros([10000])

    train_data = np.zeros([28, 28, 3, 50000])
    train_label = np.zeros([50000])

    try:
        os.makedirs(f'{metaset_dir}/mnist_bg_{index}/images/')
    except:
        print("fail to create dir")

    # testing images
    i = 0
    for j in range(size):
        sample = all_samples_test[i]

        sample_rot = random_rotation_new(sample[0])
        test_data[:, :, :, j] = creat_bg(sample_rot, img)
        test_label[j] = sample[1]
        i += 1
        # save images only for visualization; I use npy file for dataloader
        imsave(f'{metaset_dir}/mnist_bg_{index}/images/org' + str(i) + ".png", sample[0])
        imsave(f'{metaset_dir}/mnist_bg_{index}/images/syn'+ str(i) + ".png", test_data[:, :, :, j].astype(np.uint8))
    # np.save(f'{metaset_dir}/mnist_bg_{index}/test_data', test_data)
    # np.save(f'{metaset_dir}/mnist_bg_{index}/test_label', test_label)


def makeMnistbg_path(metaset_dir, img_paths, num=1):  # <fix>
    """
    Change the background of  MNIST images
    Select all testing samples from MNIST
    Store in numpy file for fast reading
    """
    index = str(num).zfill(5)
    np.random.seed(0)

    # Empty arrays
    test_data = np.zeros([28, 28, 3, 10000])
    test_label = np.zeros([10000])

    train_data = np.zeros([28, 28, 3, 50000])
    train_label = np.zeros([50000])

    try:
        os.makedirs(f'{metaset_dir}/mnist_bg_{index}/images/')  # <fix>
    except:
        None

    # testing images
    i = 0
    for j in range(size):
        file = img_paths[np.random.choice(len(img_paths), 1)]
        img = PIL.Image.open(file[0])

        sample = all_samples_test[i]
        sample_rot = random_rotation_new(sample[0])
        test_data[:, :, :, j] = creat_bg(sample_rot, img)
        test_label[j] = sample[1]
        i += 1
        # save images only for visualization; I use npy file for the dataloader
        # imsave(f'{metaset_dir}/mnist_bg_{index}/images/org' + str(i) + ".png", sample[0])
        # imsave(f'{metaset_dir}/mnist_bg_{index}/images/syn'+ str(i) + ".png", test_data[:, :, :, j].astype(np.uint8))
    np.save(f'{metaset_dir}/mnist_bg_{index}/test_data', test_data)
    np.save(f'{metaset_dir}/mnist_bg_{index}/test_label', test_label)


if __name__ == '__main__':
    # ---- load mnist images ----#
    args = parser.parse_args()
    ## we do not use the loadMnist and text-xxx.csv to load the Mnist data
    # all_samples_test = loadMnist('test', args.mnist_path+args.dataset_name.upper()+r'/')
    # all_samples_train = loadMnist('train', args.mnist_path + args.dataset_name)
    args.data = args.mnist_path
    dataset = ContrastiveLearningDataset(args)  # use the cl-model type to choose the transformation type
    all_samples_test = dataset.get_seed_dataset(args.mnist_path, args.dataset_name)  # <fix>
    all_samples_test = [
        [all_samples_test.data[:, :, i].astype(np.uint8), int(all_samples_test.targets[i])]
        for i in range(len(all_samples_test.targets))
    ]

    # ---- coco dataset - using coco training set following the  paper to rplaced the MNIST image background ----#
    coco_path = pathlib.Path(args.coco_path)
    files = sorted(list(coco_path.glob('*.jpg')) + list(coco_path.glob('*.png')) + list(coco_path.glob('*.JPEG')))

    # ---- generate sample sets ----#
    print('==== generating sample sets ====')
    num = args.metaset_size  # the number of sample sets (3000 sample sets; recommend use 200 for check the all codes)
    size = args.sampleset_size
    conut = 0

    # two ways of selecting coco images as background, both ways are similar in terms of meta set diversity
    # ----------- way 1 ----------- #
    for i in trange(num):
        try:
            img = PIL.Image.open(files[i])
            makeMnistbg(args.metaset_dir, img, conut)  # <fix>
            conut += 1
        except:
            print('jump an image!')

    # ----------- way 2 ----------- #
    # for _ in trange(num):
    #     try:
    #         b_indice = np.random.choice(len( files), 1)
    #         img_paths = np.array(files)[b_indice]
    #         makeMnistbg_path(args.metaset_dir, img_paths, conut)  # <fix>
    #         conut += 1
    #     except:
    #         print('jump an image!')
