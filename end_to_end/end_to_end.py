from cells_kitchen import config as cfg
from cells_kitchen.region_proposal.config import X_layers as rp_channels
from cells_kitchen.instance_segmentation.config import X_layers as is_channels
from cells_kitchen import utils
import numpy as np
from keras.models import load_model
import skimage.measure
import skimage.feature
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import ipdb as ipdb


def run_network(dataset, rp_model_name, is_model_name, maxima_thresh=.2, min_distance=4):

    # load data and models
    print('%s: loading data and models...' % dataset)
    data = np.load(os.path.join(cfg.data_dir, 'training_data', dataset+'.npz'), allow_pickle=True)['X'][()]
    data_rp = np.stack([data[k] for k in rp_channels], axis=-1)
    data_is = np.stack([data[k] for k in is_channels], axis=-1)
    model_rp = load_model(rp_model_name)
    model_is = load_model(is_model_name)
    sub_size = model_is.input_shape[1:3]
    print('%s: dimensions: (%i, %i)...' % (dataset, data_rp.shape[0], data_rp.shape[1]))

    # crop image if necessary
    row = data_rp.shape[0] // 16 * 16 if (data_rp.shape[0]/16)%2 != 0 else data_rp.shape[0]
    col = data_rp.shape[1] // 16 * 16 if (data_rp.shape[1] / 16) % 2 != 0 else data_rp.shape[1]

    if (row, col) != data_rp.shape:
        print('%s: cropping to dimensions: (%i, %i)...' % (dataset, row, col))
        data_rp = data_rp[:row, :col]
        data_is = data_is[:row, :col]

    # get region proposals
    print('%s: getting region proposals...' % dataset)
    rp = model_rp.predict(np.expand_dims(data_rp, 0)).squeeze()
    maxima = skimage.feature.peak_local_max(
        rp, min_distance=min_distance, threshold_abs=maxima_thresh, indices=False)
    maxima = skimage.measure.label(maxima, 8)
    maxima = skimage.measure.regionprops(maxima)
    centroids = np.array([m.centroid for m in maxima])

    # perform instance segmentation at each maximum
    print('%s: segmenting candidate neurons...' % dataset)
    subframes, segmentations, scores = [], [], []

    for m in tqdm(maxima):
        position = (int(m.centroid[0] - sub_size[0] / 2),
                    int(m.centroid[1] - sub_size[1] / 2),
                    sub_size[0],
                    sub_size[1])
        subframe = utils.get_subimg(data_is, position)
        segmentation, score = model_is.predict(subframe[None,:,:,:])
        segmentations.append(segmentation.squeeze())
        scores.append(score[0][0])
        subframes.append(utils.get_subimg(rp, position).squeeze())

    return rp, segmentations, scores, centroids, data_rp, data_is



def plot_data(dataset, rp_model_name, is_model_name, score_thresh=.2, maxima_thresh=.2, min_distance=4):

    # run network
    rp, segmentations, scores, centroids, data_rp, data_is = \
        run_network(dataset, rp_model_name, is_model_name, min_distance=min_distance, maxima_thresh=maxima_thresh)

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

    for i, s in enumerate(tqdm(segmentations)):
        if scores[i] > score_thresh:
            r, c = int(centroids[i][0] - sub_size[0] / 2), int(centroids[i][1] - sub_size[1] / 2)
            cell_map = bg.copy()
            s_temp = s
            s_temp = np.repeat(s_temp[:, :, None], 3, 2)
            cell_map[r + sub_size[0]:r + sub_size[0] * 2, c + sub_size[1]:c + sub_size[1] * 2] = s_temp
            color = cmap(np.random.rand())[:-1]
            cell_maps.append(cell_map * color)

    img = np.array(cell_maps).max(0)
    img = img[sub_size[0]:sub_size[0] + rp.shape[0], sub_size[1]:sub_size[1] + rp.shape[1]]
    y = utils.get_targets(os.path.join(cfg.data_dir, 'labels', dataset),
                    border_thickness=1, collapse_masks=True, use_curated_labels=True)['borders']
    if y.shape!=img.shape:  # trim labels if input to network was also trimmed
        y = y[:img.shape[0], :img.shape[1]]
    img = utils.add_contours(img, y)  # add ground truth cell borders
    ax[0,1].imshow(img)

    # turn off axis labels and tighten layout
    for r in range(2):
        for c in range(2):
            ax[r,c].axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(cfg.data_dir, 'results', 'e2e_figs', dataset+'.png'))


##

# settings
maxima_thresh = .3  # for finding local maxima in region proposals
score_thresh = .3  # for instance segmentation classifier
min_distance = 4
rp_model_name = r'C:\Users\erica and rick\Desktop\cells_kitchen\models\region_proposal\191006_21.09.28\unet.92-0.355555.hdf5'
is_model_name = r'C:\Users\erica and rick\Desktop\cells_kitchen\models\instance_segmentation\191008_18.56.30\segnet.221-0.275403.hdf5'

for d in cfg.datasets:
    # try:
        plot_data(d, rp_model_name, is_model_name,
                  maxima_thresh=maxima_thresh, score_thresh=score_thresh, min_distance=min_distance)
    # except:
    #     print('%s: ANALYSIS FAILED! WTF!!!' % d)

