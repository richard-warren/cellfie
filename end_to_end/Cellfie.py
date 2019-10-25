from cellfie import utils
import numpy as np
from keras.models import load_model
import skimage.measure
import skimage.feature
from tqdm import tqdm
import os
import yaml
import matplotlib.pyplot as plt
import ipdb


class Cellfie:

    def __init__(self, rp_model, is_model):

        # initializations
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        self.rp_model = rp_model
        self.is_model = is_model

        # load models
        print('loading region proposal model...')
        self.model_rp = load_model(rp_model)
        print('loading instance segmentation model...')
        self.model_is = load_model(is_model)
        self.sub_size = self.model_is.input_shape[1:3]

    def analyze_data(self, input_data, maxima_thresh=.1, min_distance=4, use_mean_mask=True, mask_scaling=1.5):

        # load data
        print('loading input data...')
        self.data = np.load(input_data, allow_pickle=True)['X'][()]

        with open(os.path.join(os.path.split(self.rp_model)[0], 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        data_rp = np.stack([self.data[k] for k in cfg['X_layers']], axis=-1)

        with open(os.path.join(os.path.split(self.is_model)[0], 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        data_is = np.stack([self.data[k] for k in cfg['X_layers']], axis=-1)


        # crop image if necessary
        row = data_rp.shape[0] // 16 * 16 if (data_rp.shape[0] / 16) % 2 != 0 else data_rp.shape[0]
        col = data_rp.shape[1] // 16 * 16 if (data_rp.shape[1] / 16) % 2 != 0 else data_rp.shape[1]
        if (row, col) != data_rp.shape:
            print('cropping to dimensions: (%i, %i)...' % (row, col))
            data_rp = data_rp[:row, :col]
            data_is = data_is[:row, :col]

        # get region proposals
        print('getting region proposals...')
        self.region_proposals = self.model_rp.predict(np.expand_dims(data_rp, 0)).squeeze()
        maxima = skimage.feature.peak_local_max(
            self.region_proposals, min_distance=min_distance, threshold_abs=maxima_thresh, indices=False)
        maxima = skimage.measure.label(maxima, 8)
        maxima = skimage.measure.regionprops(maxima)
        self.centroids = np.array([m.centroid for m in maxima])

        # perform instance segmentation at each maximum
        print('segmenting candidate neurons...')
        self.segmentations, self.scores, self.subframes = [], [], []

        for m in tqdm(maxima):
            position = (int(m.centroid[0] - self.sub_size[0] / 2),
                        int(m.centroid[1] - self.sub_size[1] / 2),
                        self.sub_size[0],
                        self.sub_size[1])
            subframe = utils.get_subimg(data_is, position, padding='median_local')
            segmentation, score = self.model_is.predict(subframe[None, :, :, :])
            self.segmentations.append(segmentation.squeeze())
            self.scores.append(score[0][0])
            self.subframes.append(subframe)

        self.compute_full_masks(use_mean_mask, mask_scaling)
        self.plot_fig()

    def compute_full_masks(self, use_mean_mask=True, mask_scaling=1.5):

        # make padded zeros into which subframe segmentations will be placed
        dims = (self.region_proposals.shape[0] + 2 * self.sub_size[0],
                self.region_proposals.shape[1] + 2 * self.sub_size[1])
        padded_bg = np.zeros(dims)
        sub_size = self.sub_size

        # create neuron mask based on mean neuron
        if use_mean_mask:
            mask = np.mean(self.segmentations, 0)
            mask = skimage.transform.resize(
                mask, (sub_size[0]*mask_scaling, sub_size[1]*mask_scaling), mode='constant')  # increase size of mask
            r, c = (round(mask.shape[d] / 2 - self.sub_size[d] / 2) for d in (0, 1))
            mask = mask[r:r + sub_size[0], c:c + sub_size[1]]  # crop out the middle of the mask
            mask = mask - mask.min()
            mask = mask / mask.max()
        else:
            mask = np.ones(sub_size)

        self.cell_masks = []
        for i, s in enumerate(tqdm(self.segmentations)):
            r, c = int(self.centroids[i][0] - sub_size[0] / 2), int(self.centroids[i][1] - sub_size[1] / 2)
            cell_map = padded_bg.copy()
            cell_seg = s.copy() * mask
            cell_map[r + sub_size[0]:r + sub_size[0] * 2, c + sub_size[1]:c + sub_size[1] * 2] = cell_seg
            cell_map = cell_map[sub_size[0]:sub_size[0] + self.region_proposals.shape[0], sub_size[1]:sub_size[1] + self.region_proposals.shape[1]]  # restrict to original dimensions
            self.cell_masks.append(cell_map)

    def plot_fig(self, score_thresh=.2):

        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 8))
        cmap = plt.get_cmap('gist_rainbow')

        # show input data
        ax[0].imshow(self.data['corr'], cmap='gray')

        # show region proposal with maxima scatter
        ax[1].imshow(self.region_proposals, cmap='gray')
        bins = np.array(self.scores) > score_thresh
        ax[1].scatter(self.centroids[bins, 1], self.centroids[bins, 0], 3, c=np.array(self.scores)[bins],
                         cmap=plt.get_cmap('rainbow'))

        # show final instance segmentation
        imgs = []
        for i, s in enumerate(self.cell_masks):
            if self.scores[i] > score_thresh:
                imgs.append(np.repeat(s[:, :, None], 3, 2) * cmap(np.random.rand())[:-1])
        img = np.array(imgs).max(0)
        ax[2].imshow(img)




