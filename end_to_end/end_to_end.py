from cellfie import utils
import numpy as np
from keras.models import load_model
import skimage.measure
import skimage.feature
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import ipdb
import yaml


class Cellfie:

    def __init__(self, input_data, rp_model, is_model,
                 maxima_thresh=.1, min_distance=4):

        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        self.input_data = input_data
        self.rp_model = rp_model
        self.is_model = is_model

        self.maxima_thresh = maxima_thresh  # for finding local maxima in region proposals
        self.min_distance = min_distance

        # load data
        print('loading input data...')
        data = np.load(input_data, allow_pickle=True)['X'][()]

        # load region proposal model
        print('loading region proposal model...')
        with open(os.path.join(os.path.split(rp_model)[0], 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        data_rp = np.stack([data[k] for k in cfg['X_layers']], axis=-1)
        model_rp = load_model(rp_model)

        # load instance segmentation model
        print('loading instance segmentation model...')
        with open(os.path.join(os.path.split(is_model)[0], 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        data_is = np.stack([data[k] for k in cfg['X_layers']], axis=-1)
        model_is = load_model(is_model)
        sub_size = model_is.input_shape[1:3]

        # crop image if necessary
        row = data_rp.shape[0] // 16 * 16 if (data_rp.shape[0] / 16) % 2 != 0 else data_rp.shape[0]
        col = data_rp.shape[1] // 16 * 16 if (data_rp.shape[1] / 16) % 2 != 0 else data_rp.shape[1]
        if (row, col) != data_rp.shape:
            print('cropping to dimensions: (%i, %i)...' % (row, col))
            data_rp = data_rp[:row, :col]
            data_is = data_is[:row, :col]

        # get region proposals
        print('getting region proposals...')
        self.region_proposals = model_rp.predict(np.expand_dims(data_rp, 0)).squeeze()
        maxima = skimage.feature.peak_local_max(
            self.region_proposals, min_distance=min_distance, threshold_abs=maxima_thresh, indices=False)
        maxima = skimage.measure.label(maxima, 8)
        maxima = skimage.measure.regionprops(maxima)
        self.centroids = np.array([m.centroid for m in maxima])

        # perform instance segmentation at each maximum
        print('segmenting candidate neurons...')
        self.segmentations, self.scores, self.subframes = [], [], []

        for m in tqdm(maxima):
            position = (int(m.centroid[0] - sub_size[0] / 2),
                        int(m.centroid[1] - sub_size[1] / 2),
                        sub_size[0],
                        sub_size[1])
            subframe = utils.get_subimg(data_is, position, padding='median_local')
            segmentation, score = model_is.predict(subframe[None, :, :, :])
            self.segmentations.append(segmentation.squeeze())
            self.scores.append(score[0][0])
            self.subframes.append(subframe)





##
def plot_data(dataset, rp_model_name, is_model_name, rp_channels, is_channels,
              score_thresh=.2, maxima_thresh=.2, min_distance=4, use_mask=False, add_ground_truth=True):

    # run network
    rp, segmentations, scores, centroids, data_rp, data_is, subframes = \
        segment(dataset, rp_model_name, is_model_name, rp_channels, is_channels,
                min_distance=min_distance, maxima_thresh=maxima_thresh)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 9))

    # show region proposals, and first two channels of input data
    ax[0, 0].imshow(rp, cmap='gray')
    bins = np.array(scores) > maxima_thresh
    ax[0, 0].scatter(centroids[bins, 1], centroids[bins, 0], 3, c=np.array(scores)[bins], cmap=plt.get_cmap('rainbow'))
    ax[1, 0].imshow(data_rp[:, :, 0], cmap='gray')
    ax[1, 1].imshow(data_rp[:, :, 1], cmap='gray')

    # add cells in different colors
    print('%s: creating image with colored neurons...' % dataset)
    sub_size = segmentations[0].shape
    cmap = plt.get_cmap('gist_rainbow')
    bg = np.zeros((rp.shape[0] + 2 * sub_size[0], rp.shape[1] + 2 * sub_size[1], 3))
    cell_maps = []

    # create neuron mask based on mean neuron
    if use_mask:
        scaling = 1.5
        mask = np.mean(segmentations, 0)
        mask = skimage.transform.resize(mask, (sub_size[0]*scaling, sub_size[1]*scaling), mode='constant')
        r, c = (round(mask.shape[d]/2-sub_size[d]/2) for d in (0, 1))
        mask = mask[r:r+sub_size[0], c:c+sub_size[1]]
        mask = mask - mask.min()
        mask = mask / mask.max()
    else:
        mask = np.ones(sub_size)

    for i, s in enumerate(tqdm(segmentations)):
        if scores[i] > score_thresh:
            r, c = int(centroids[i][0] - sub_size[0] / 2), int(centroids[i][1] - sub_size[1] / 2)
            cell_map = bg.copy()
            cell_seg = s.copy() * mask
            cell_seg = np.repeat(cell_seg[:, :, None], 3, 2)
            cell_map[r + sub_size[0]:r + sub_size[0] * 2, c + sub_size[1]:c + sub_size[1] * 2] = cell_seg
            color = cmap(np.random.rand())[:-1]
            cell_map = cell_map * color

            # # uncomment to put input data in the summary image instead of network segmentation (for debugging purposes)
            # cell_map[r + sub_size[0]:r + sub_size[0] * 2, c + sub_size[1]:c + sub_size[1] * 2] = \
            #     np.repeat(subframes[i][:,:,None], 3, 2)

            cell_maps.append(cell_map)

    img = np.array(cell_maps).max(0)
    img = img[sub_size[0]:sub_size[0] + rp.shape[0], sub_size[1]:sub_size[1] + rp.shape[1]]

    if add_ground_truth:
        y = utils.get_targets(os.path.join(cfg.data_dir, 'labels', dataset),
                        border_thickness=1, collapse_masks=True, use_curated_labels=True)['borders']
        if y.shape != img.shape:  # trim labels if input to network was also trimmed
            y = y[:img.shape[0], :img.shape[1]]
        img = utils.add_contours(img, y)  # add ground truth cell borders

    ax[0,1].imshow(img)

    # turn off axis labels and tighten layout
    for r in range(2):
        for c in range(2):
            ax[r,c].axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(cfg.data_dir, 'results', 'e2e_figs', dataset+'.png'))
