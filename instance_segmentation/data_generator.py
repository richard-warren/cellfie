from cells_kitchen.config import data_dir
from cells_kitchen.instance_segmentation import config as cfg
from cells_kitchen import utils
from keras.utils import Sequence
import numpy as np
import pandas as pd
import os
import cv2
import ipdb
from scipy.ndimage.measurements import center_of_mass
import cv2


class DataGenerator(Sequence):
    '''
    each call returns stack of X and ys, whers stack contains batch_size images randomly selected for datasets
    each frame will be square subframe of random size, but no larger than smallest dataset
    locations in frame will also be random
    '''

    def __init__(self, datasets, batch_size=8, subframe_size=(40, 40), epoch_size=64, rotation=True, scaling=(1, 1),
                 fraction_positive_egs=.5, jitter=2, negative_eg_distance=8):

        # initialization
        self.datasets = datasets
        self.batch_size = batch_size
        self.subframe_size = subframe_size
        self.epoch_size = epoch_size
        self.rotation = rotation
        self.scaling = scaling
        self.fraction_positive_egs = fraction_positive_egs
        self.jitter = jitter
        self.negative_eg_distance = negative_eg_distance

        # load features and labels into DataFrame
        self.data = pd.DataFrame(index=datasets, columns=['X', 'y', 'negative_eg_inds'])
        for d in datasets:

            data_sub = np.load(os.path.join(data_dir, 'training_data', d + '.npz'), allow_pickle=True)
            X = np.stack([data_sub['X'][()][k] for k in cfg.X_layers], axis=2)
            y = data_sub['neuron_masks']

            # get logical masks representing pixels where negative egs can be centered at
            # todo: the following is the shittiest code ever // need to make this faster, maybe circle convolution
            negative_eg_mask = np.zeros(X.shape[:2], dtype='uint8')
            for n in range(y.shape[0]):  # loop across neurons
                center = np.round(center_of_mass(y[n])).astype('int')
                negative_eg_mask = cv2.circle(
                    negative_eg_mask, (center[1], center[0]), negative_eg_distance, 1, thickness=-1)
            negative_eg_mask = np.invert(negative_eg_mask.astype('bool'))
            negative_eg_inds = np.transpose(np.stack(np.where(negative_eg_mask)))

            self.data.loc[d, :] = (X, y, negative_eg_inds)

        self.shape_X = (batch_size,) + subframe_size + (self.data.loc[datasets[0], 'X'].shape[-1],)  # batch size X height X width X depth
        self.shape_y = (batch_size,) + subframe_size

    def __getitem__(self, index):

        # gets data for batch
        X = np.zeros(self.shape_X)
        y = np.zeros(self.shape_y)  # neuron mask
        is_neuron = np.zeros(self.batch_size, dtype='bool')

        for i in range(self.batch_size):

            # select random dataset, eg type, and neuron if positive eg
            dataset_ind = np.random.randint(len(self.datasets))
            is_neuron[i] = self.fraction_positive_egs > np.random.uniform(0, 1)  # decide whether this is a positive or negative example

            # get subframe scaling
            if self.scaling:
                scale = np.random.uniform(2 - self.scaling[1], 2 - self.scaling[0])  # the 2 minus is bc taking a BIGGER subframe results in a SMALLER resized subframe
            else:
                scale = 1
            dx = np.floor(self.subframe_size[1] * scale / 2).astype('uint8')  # dx and xy are half the width of subframe
            dy = np.floor(self.subframe_size[0] * scale / 2).astype('uint8')


            # pick subframe center
            if is_neuron[i]:
                neuron_ind = np.random.randint(self.data.y[dataset_ind].shape[0])
                center = np.round(center_of_mass(self.data.y[dataset_ind][neuron_ind])).astype('int')
                jitter = np.random.randint(-self.jitter, self.jitter+1, size=2)
                center = center + jitter
            else:
                ind = np.random.randint(self.data.negative_eg_inds[dataset_ind].shape[0])
                center = self.data.negative_eg_inds[dataset_ind][ind]

            # get subframe
            position = (center[0]-dy, center[1]-dx, self.subframe_size[0], self.subframe_size[1])
            X_temp = utils.get_subimg(self.data.X[dataset_ind], position, padding='median_local')

            if is_neuron[i]:
                y_temp = utils.get_subimg(self.data.y[dataset_ind][neuron_ind], position, padding='median_local')
            else:
                y_temp = np.zeros(self.subframe_size)

            if self.scaling:
                X_temp = cv2.resize(X_temp, self.subframe_size)
                y_temp = cv2.resize(y_temp.astype('uint8'), self.subframe_size).astype('bool')

            X[i] = X_temp
            y[i] = y_temp

        # rotate
        if self.rotation:
            rotations = np.random.randint(0, 4)  # number of 90 degree rotations to perform
            if rotations:
                X = np.rot90(X, rotations, axes=(1, 2))
                y = np.rot90(y, rotations, axes=(1, 2))

        y = np.expand_dims(y, -1)  # a temporary hack because keras expects multiple output channels
        return X, [y, is_neuron], [is_neuron, np.ones(is_neuron.shape)]  # third output are sample weights // use is_neuron for sample wiehgts for mask, bc we ignore negative examples in the mask backprop

    def __len__(self):
        return self.epoch_size
