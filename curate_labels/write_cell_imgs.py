from cells_kitchen import config as cfg
from cells_kitchen import utils
import os
import numpy as np
from tqdm import tqdm
import imageio
from skimage.transform import resize
import time

# settings
datasets = ['N.04.00.t', 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'YST', 'K53', 'J115', 'J123']
channels = ['corr', 'median']
contrast = [1, 99.9]
crop = 100
scaling = 4


for dataset in datasets:

    print('%s: writing cell images...' % dataset)

    # load summary images
    summary_imgs = np.load(os.path.join(cfg.data_dir, 'training_data', 'caiman', dataset + '.npz'), allow_pickle=True)
    X = [utils.enhance_contrast(summary_imgs['X'][()][k], contrast) for k in channels]
    height, width = X[0].shape[0], X[0].shape[1]

    # compute borders
    borders = utils.get_targets(os.path.join(cfg.data_dir, 'caiman', 'labels', dataset), border_thickness=1)['borders']

    for cell in tqdm(range(borders.shape[0])):

        # find crop position
        center = np.argwhere(borders[cell]).mean(axis=0).astype('int')
        ylims = slice(max(center[0] - crop // 2, 0), min(center[0] + crop // 2, height))
        xlims = slice(max(center[1] - crop // 2, 0), min(center[1] + crop // 2, width))

        # collect and save image
        img = np.concatenate([utils.add_contours(x[ylims, xlims], borders[cell][ylims, xlims]) for x in X], axis=1)
        img = resize(img, (img.shape[0]*scaling, img.shape[1]*scaling, 3), mode='constant')
        img = (img * 255).astype('uint8')
        imageio.imwrite(os.path.join(cfg.data_dir, 'caiman', 'labels', dataset, 'images', 'cell_'+str(cell)+'.png'), img)
