'''
make sample images for the github repo readme
'''

from cellfie import utils
from cellfie.end_to_end.Cellfie import segment
import numpy as np
from PIL import Image
from skimage.transform import resize, rescale
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage.draw

# SEGMENT DATASET

# settings
dataset = 'K53'
channels = ['corr', 'median', 'std']
output_folder = 'images'
rp_model_name = r'C:\Users\erica and rick\Desktop\cellfie\models\region_proposal\train_test_same\unet.988-0.124001.hdf5'  # train test same
is_model_name = r'C:\Users\erica and rick\Desktop\cellfie\models\instance_segmentation\traintestsame\segnet.96-0.265353.hdf5'  # train test same
maxima_thresh = .1  # for finding local maxima in region proposals
min_distance = 4


rp, segmentations, scores, centroids, data_rp, data_is, subframes = \
segment(dataset, rp_model_name, is_model_name, channels, channels,
    min_distance=min_distance, maxima_thresh=maxima_thresh)


# stacks images, where dim 2 is the image dimension, and 0,1 are row, col dimensions
def stack_images(imgs, offset, contrast=(0,99)):
    n = imgs.shape[2]
    stack = np.ones((imgs.shape[0]+offset*(n-1), imgs.shape[1]+offset*(n-1)))
    for i in reversed(range(n)):
        r, c = (n-i-1)*offset, i*offset
        stack[r:r+imgs.shape[0], c:c+imgs.shape[1]] = utils.enhance_contrast(imgs[:,:,i], contrast)

    return stack

# saves images with input images stacked on left, and output image on right
def save_inout_img(input, output, file_name, hgt_out=300, separation=.2, stack_separation=.1):

    stack = stack_images(input, int(output.shape[0]*stack_separation))
    hgt, wid = stack.shape[0], int(stack.shape[1] * (2+separation))
    output = resize(output.copy(), (hgt, int(output.shape[0]*(hgt/output.shape[0]))), mode='constant', cval=1)
    output = utils.enhance_contrast(output, [5, 99])

    rp_sample = np.ones((hgt, wid))
    rp_sample[:, :stack.shape[1]] = stack
    rp_sample[:, -output.shape[1]:] = output
    rp_sample = rescale(rp_sample, (hgt_out/hgt), mode='constant', cval=1)

    img = Image.fromarray((rp_sample*255).astype('uint8'))
    img.save(file_name)


## make region proposal and instance segmentation sample images

cells = [21, 22, 28]
save_inout_img(data_rp, rp, os.path.join(output_folder, 'rp_sample_noarrow.png'))
for cell in cells:
    save_inout_img(subframes[cell], segmentations[cell], os.path.join(output_folder, 'is_sample%i.png'%cell),
                   hgt_out = int(rp.shape[0]/len(cells)), separation=.5)


## make images for gif

score_thresh = .3  # for instance segmentation classifier

# create blank image to start
final_hgt = 300
sub_size = segmentations[0].shape
bg = np.zeros((data_rp.shape[0] + 2 * sub_size[0], rp.shape[1] + 2 * sub_size[1], 3))
cmap = plt.get_cmap('gist_rainbow')

# create neuron mask
scaling = 1.5
mask = np.mean(segmentations, 0)
mask = resize(mask, (sub_size[0]*scaling, sub_size[1]*scaling), mode='constant')
r, c = (round(mask.shape[d]/2-sub_size[d]/2) for d in (0, 1))
mask = mask[r:r+sub_size[0], c:c+sub_size[1]]
mask = mask - mask.min()
mask = mask / mask.max()

prev_map = bg
count = 0
input = data_rp[:,:,0]
input = np.repeat(input[:,:,None],3,2)  # add color dimension
num_neurons = len(segmentations)

for i in tqdm(range(num_neurons)):
    if scores[i] > score_thresh:

        # get input image and highlight subframe
        r, c = int(centroids[i][0] - sub_size[0] / 2), int(centroids[i][1] - sub_size[1] / 2)
        rp_img = rp.copy()
        rp_img = np.repeat(rp_img[:,:,None],3,2)  # add color dimension
        rp_mask = np.ones(rp_img.shape) * .3
        rp_mask[max(0, r) : min(r+sub_size[0], rp_img.shape[0]), max(0, c) : min(c+sub_size[1], rp_img.shape[1])] = 1
        rp_img = rp_img * rp_mask

        # add next colored segmentation
        cell_map = bg.copy()
        cell_seg = segmentations[i].copy() * mask
        cell_seg = np.repeat(cell_seg[:, :, None], 3, 2)
        cell_map[r + sub_size[0]:r + sub_size[0] * 2, c + sub_size[1]:c + sub_size[1] * 2] = cell_seg
        color = cmap(np.random.rand())[:-1]
        cell_map = cell_map * color

        # save current image
        img = np.array([prev_map, cell_map]).max(0)
        prev_map = img
        img = img[sub_size[0]:sub_size[0] + rp.shape[0], sub_size[1]:sub_size[1] + rp.shape[1]]
        weights = np.repeat(img.max(2)[:,:,None],3,2)

        input_temp = input.copy()
        input_temp = utils.scale_img(input_temp*img + (1-weights)*input_temp)

        img = np.concatenate((input_temp, rp_img, img), axis=1)
        img = rescale(img, (final_hgt/img.shape[0]), mode='constant')
        img = Image.fromarray((img * 255).astype('uint8'))
        img.save(os.path.join(output_folder, 'gif_imgs', 'img%04d.png' % count))
        count += 1


##
ax.clear()
new = input*img + (1-input)*input
plt.imshow(new)












