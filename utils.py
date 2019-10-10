import glob
import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import json
import cv2
import tifffile
from PIL import Image, ImageDraw, ImageFont
from cells_kitchen import config as cfg
from scipy import signal


def get_frames(folder, frame_inds=0, frame_num=False):
    """
    gets stack of images from folder containing tiff files // if frame_inds is given, these are the frames
    included in stack // otherwise, frame_num evenly spaced images are returned in the stack
    """

    files = glob.glob(os.path.join(folder, '*.tif*'))  # works for .tif AND .tiff files
    if frame_num:
        frame_num = min(len(files), frame_num, 1500)
        frame_inds = np.floor(np.linspace(0, len(files)-1, frame_num)).astype('int16')
    imgs = tifffile.imread(np.array(files)[frame_inds].tolist()).astype('float32')

    return imgs


def get_subimg(img_in, position, padding='mean'):
    '''
    given an image and sub image position (row, col, height, width), where row and col are positions of top left corner,
    returns a cropped subframe // allows subframes to be requested that are out of the bounds of the image,
    and it pads the image to compensate, either with zeros (padding='zeros'), or with the mean of the
    image (padding='mean')
    '''

    row, col, hgt, wid = position[0], position[1], position[2], position[3]
    img = np.expand_dims(img_in.copy(), 2) if len(img_in.shape)==2 else img_in.copy()  # add color dimension if necessary

    # if subframe fits within frame
    if row>=0 and col>=0 and (row+hgt)<img.shape[0] and (col+wid)<img.shape[1]:
        subimg = img[row:row+hgt, col:col+wid]
        return subimg.squeeze()

    # if subframe contains ANY of the frame
    elif row>(-hgt) and col>(-wid) and row<img.shape[0] and col<img.shape[1]:

        img_expanded = np.ones((img.shape[0]+2*hgt, img.shape[1]+2*wid, img.shape[2]), dtype=img.dtype)
        if padding == 'zeros':
            multiplier = 0
        elif padding == 'mean':
            multiplier = np.mean(img, axis=(0, 1))
        elif padding == 'mean_local':
            r, c = max(row, 0), max(col, 0)
            img_local = img[r:min(r + hgt, img.shape[0]), c:min(c + wid, img.shape[1])]
            multiplier = np.mean(img_local, axis=(0, 1))
        elif padding == 'median':
            multiplier = np.median(img, axis=(0, 1))
        elif padding == 'median_local':
            r, c = max(row, 0), max(col,0)
            img_local = img[r:min(r+hgt, img.shape[0]), c:min(c+wid, img.shape[1])]
            multiplier = np.median(img_local, axis=(0, 1))

        img_expanded = img_expanded * multiplier
        img_expanded[hgt:hgt+img.shape[0], wid:wid+img.shape[1]] = img
        subimg = img_expanded[row+hgt:row+2*hgt, col+wid:col+2*wid]

        return subimg.squeeze()

    # if frame is not contained within requested subframe, return an error
    else:
        print('WARNING: Requested sub image is completely outside of image dimensions')
        return -1


def get_correlation_image(imgs):
    """
    given stack of images, returns image representing temporal correlation between each pixel and surrounding eight pixels
    """

    # define kernel
    # kernel = np.ones((3, 3), dtype='float32')
    # kernel[1, 1] = 0

    # kernel = np.array([[0,1,0], [1,0,1], [0,1,0]], dtype='float32')

    kernel = signal.gaussian(7, 3, sym=True)
    kernel = np.outer(kernel, kernel)
    kernel[3, 3] = 0

    mask = convolve(np.ones(imgs.shape[1:], dtype='float32'), kernel, mode='constant')

    # normalize image
    imgs_norm = imgs.copy()
    imgs_norm -= np.mean(imgs_norm, axis=0)
    imgs_std = np.std(imgs_norm, axis=0)
    imgs_std[imgs_std == 0] = np.inf
    imgs_norm /= imgs_std

    # compute correlation image
    img_corr = convolve(imgs_norm, kernel[np.newaxis, :], mode='constant') / mask
    img_corr = imgs_norm * img_corr
    img_corr = np.mean(img_corr, 0)

    return img_corr


def scale_img(img):
    """ scales numpy array between 0 and 1"""

    img_scaled = img.copy()
    if np.ptp(img_scaled):
        img_scaled = (img_scaled - np.min(img_scaled)) / np.ptp(img_scaled)
    return img_scaled


def get_targets(folder, collapse_masks=False, centroid_radius=2, border_thickness=2, use_curated_labels=False):
    """
    for folder containing labeled data, returns masks for soma, border of cells, and centroid. returned as 3D bool
    stacks, with one mask per cell, unless collapse_masks is True, in which case max is taken across all cells
    """

    # get image dimensions
    with open(os.path.join(folder, 'info.json')) as f:
        dimensions = json.load(f)['dimensions'][1:3]

    # load labels
    if use_curated_labels and os.path.exists(os.path.join(folder, 'regions', 'consensus_regions_curated.json')):
        print('using curated labels for %s...' % folder)
        file = os.path.join('regions', 'consensus_regions_curated.json')
    else:
        file = os.path.join('regions', 'consensus_regions.json')
    with open(os.path.join(folder, file)) as f:
        cell_masks = [np.array(x['coordinates']) for x in json.load(f)]

    # compute masks for each neuron
    masks_soma, masks_border, masks_centroids = \
        [np.zeros((len(cell_masks), dimensions[0], dimensions[1]), dtype=bool) for _ in range(3)]

    for i, cell in enumerate(cell_masks):
        masks_soma[i, cell[:, 0], cell[:, 1]] = True
        _, contour, _ = cv2.findContours(masks_soma[i].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        masks_border[i] = cv2.drawContours(np.zeros(dimensions), contour, 0, 1, thickness=border_thickness).astype('bool')
        center = np.mean(cell, 0).astype('int')
        masks_centroids[i] = cv2.circle(masks_centroids[i].astype('uint8'), (center[1], center[0]), centroid_radius, 1, thickness=-1)

    # collapse across neurons
    if collapse_masks:
        [masks_soma, masks_border, masks_centroids] = \
            [np.max(x, 0) for x in (masks_soma, masks_border, masks_centroids)]

    masks = {
        'somas': masks_soma,
        'borders': masks_border,
        'centroids': masks_centroids
    }

    return masks


def add_contours(img, contour, color=(1, 0, 0)):
    """given 2D img, and 2D contours, returns 3D image with contours added in color"""

    img_contour = img.copy()

    if len(img_contour.shape)==2:
        img_contour = np.repeat(img_contour[:, :, np.newaxis], 3, axis=2)

    inds = np.argwhere(contour)
    img_contour[inds[:, 0], inds[:, 1], :] = np.array(color)

    return img_contour


def enhance_contrast(img, percentiles=(5, 95)):
    """given 2D image, rescales the image between lower and upper percentile limits"""

    img = img.copy()
    limits = np.percentile(img.flatten(), percentiles)
    img_enhanced = np.clip(img-limits[0], 0, limits[1]-limits[0]) / np.ptp(limits)
    # img = np.clip(img, limits[0], limits[1]) / np.ptp(limits)

    return img_enhanced


def save_prediction_img(file, X, y, y_pred=None, height=800, X_contrast=(0,100), column_titles=None):
    """ given X and y_pred for a single image, (network output), writes an image to file concatening everybody """

    # scaled from 0->1
    for i in range(X.shape[-1]):
        X[:, :, i] = scale_img(X[:, :, i])
        if X_contrast != (0, 100):
            X[:, :, i] = enhance_contrast(X[:, :, i], percentiles=X_contrast)
    if type(y_pred) == np.ndarray:
        for i in range(y_pred.shape[-1]):
            y_pred[:, :, i] = scale_img(y_pred[:, :, i])

    # concatenate layers horizontally
    X_cat = np.reshape(X, (X.shape[0], -1), order='F')
    y_cat = np.reshape(y, (y.shape[0], -1), order='F')
    if type(y_pred)==np.ndarray:
        y_pred_cat = np.reshape(y_pred, (y_pred.shape[0], -1), order='F')

    # make image where three rows are X, y, and y_pred
    if type(y_pred)==np.ndarray:
        cat = np.zeros((X.shape[0]*3, max(X_cat.shape[1], y_cat.shape[1])))
    else:
        cat = np.zeros((X.shape[0] * 2, max(X_cat.shape[1], y_cat.shape[1])))
    cat[:X.shape[0], :X_cat.shape[1]] = X_cat
    cat[X.shape[0]:X.shape[0]*2, :y_cat.shape[1]] = y_cat
    if type(y_pred)==np.ndarray:
        cat[X.shape[0]*2:, :y_cat.shape[1]] = y_pred_cat

    img = Image.fromarray((cat * 255).astype('uint8'))

    if column_titles is not None:
        font = ImageFont.truetype("arial.ttf", 20)
        img_draw = ImageDraw.Draw(img)
        for i, t in enumerate(column_titles):
            img_draw.text((X.shape[1] * i, 0), t, fill=255, font=font)

    img = img.resize((int((cat.shape[1] / cat.shape[0]) * height), height), resample=Image.NEAREST)
    img.save(file)


def write_sample_imgs(X_contrast=(0,100)):
    '''writes sample images for training and test data for all .npz files in training_data folder'''

    print('writing sample summary images to disk...')
    files = glob.glob(os.path.join(cfg.data_dir, 'training_data', '*.npz'))

    for f in files:
        data = np.load(f)
        X_mat = np.stack(data['X'][()].values(), axis=2)
        y_mat = np.stack(data['y'][()].values(), axis=2)
        file_name = os.path.join(cfg.data_dir, 'training_data', os.path.splitext(f)[0] + '.png')
        save_prediction_img(file_name, X_mat, y_mat, X_contrast=X_contrast, column_titles=data['X'][()].keys())


def write_sample_border_imgs(channels=['corr'], height=800, contrast=(0,100)):

    print('writing sample summary images with borders to disk...')
    files = glob.glob(os.path.join(cfg.data_dir, 'training_data', '*.npz'))

    for f in files:

        # load data
        dataset = os.path.splitext(os.path.basename(f))[0]  # get dataset name
        data = np.load(f)
        X = data['X'][()]

        # restrict to requested channels, and borders only for y
        X = dict((k, X[k]) for k in channels)  # restrict to requested channels
        X = np.stack(X.values(), axis=2)

        y = get_targets(os.path.join(cfg.data_dir, 'labels', dataset),
                        border_thickness=1, collapse_masks=True, use_curated_labels=cfg.use_curated_labels)['borders']

        # add borders
        img = np.zeros((y.shape[0], y.shape[1]*len(channels), 3))
        for i in range(len(channels)):
            temp = enhance_contrast(X[:,:,i], percentiles=contrast)
            img[:, i*(X.shape[1]):(i+1)*X.shape[1], :] = add_contours(temp, y)

        file_name = os.path.join(cfg.data_dir, 'training_data', os.path.splitext(f)[0] + '_borders.png')

        img = Image.fromarray((img * 255).astype('uint8'))
        img = img.resize((int((img.width / img.height) * height), height), resample=Image.NEAREST)
        img.save(file_name)

    print('all done!')







