import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import argparse  # <fix>
from matplotlib.pyplot import MultipleLocator
import random

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--save-dir', metavar='DIR', default='/data/pengru/Contrastive_AutoEval/checkpoints/',
                    help='path to save checkpoints')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--meta-dataset-name', default='mnist',
                    help='meta dataset name', choices=['MNIST', 'CIFAR-10', 'CIFAR-100', 'COCO', 'TinyImageNet'])
parser.add_argument('--indice', default=0, type=int, metavar='N',
                    help='the color index of color palate')
# parser.add_argument('--xlim', default='(15, 65)', metavar='B',
#                     help='X Axis range of scatter plot')
# parser.add_argument('--ylim', default='(10, 90)', metavar='B',
#                     help='Y Axis range of scatter plot')

def main():
    args = parser.parse_args()

    cla_acc = np.load(f'{args.save_dir}/accuracy_cla{args.epochs}.npy')
    con_acc = np.load(f'{args.save_dir}/accuracy_con{args.epochs}.npy')

    majorFormatter = FormatStrFormatter('%0.1f')

    palette = sns.color_palette("bright")
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30

    sns.set()
    sns.set(font_scale=1.3)
    sns.set_style('ticks', {'axes.facecolor': '0.96', 'axes.linewidth': 20, 'axes.edgecolor': '0.15'})

    f, ax1 = plt.subplots(1, 1, tight_layout=True)
    ax1.tick_params(which='major', direction="in")
    sns.regplot(ax=ax1, color=palette[args.indice], x=con_acc, y=cla_acc, robust=True, scatter_kws={'alpha': 0.5}, # , 's': 20
                label='{:<2}\n{:>1}\n{:>1}'.format(args.meta_dataset_name, r'$r$' + '={:.3f}'.format(stats.pearsonr(con_acc, cla_acc)[0]),
                                                   chr(961) + '={:.3f}'.format(stats.spearmanr(con_acc, cla_acc)[0])))
    ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=0, borderpad=0.5, markerscale=2,
               prop={'weight': 'medium', 'size': '16'})

    # set title of the x and y axes
    ax1.set_ylabel('Classification accuracy (%)', fontsize=15)
    ax1.set_xlabel('Contrast accuracy (%)', fontsize=15)

    ax1.xaxis.set_major_formatter(majorFormatter)
    ax1.yaxis.set_major_formatter(majorFormatter)
    f.savefig(f'{args.save_dir}/correlation.pdf', bbox_inches='tight', pad_inches=0.0)


if __name__ == "__main__":
    main()