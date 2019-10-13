'''
make sample images for the github repo readme
'''

from cellfie import config as cfg
from cellfie import utils
import numpy as np
from keras.models import load_model
import skimage.measure
import skimage.feature
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os



## initializations

# settings
dataset = 'K53'
channels = ['corr', 'median', 'std']
output_folder = 'images'
rp_model_name = r'C:\Users\erica and rick\Desktop\cellfie\models\region_proposal\train_test_same\unet.988-0.124001.hdf5'  # train test same
is_model_name = r'C:\Users\erica and rick\Desktop\cellfie\models\instance_segmentation\traintestsame\segnet.96-0.265353.hdf5'  # train test same
maxima_thresh = .1  # for finding local maxima in region proposals
score_thresh = .3  # for instance segmentation classifier
min_distance = 4


print('%s: loading data and models...' % dataset)
data = np.load(os.path.join(cfg.data_dir, 'training_data', dataset+'.npz'), allow_pickle=True)['X'][()]
data = np.stack([data[k] for k in channels], axis=-1)
model_rp = load_model(rp_model_name)
model_is = load_model(is_model_name)
sub_size = model_is.input_shape[1:3]

# crop image if necessary
row = data.shape[0] // 16 * 16 if (data.shape[0]/16)%2 != 0 else data.shape[0]
col = data.shape[1] // 16 * 16 if (data.shape[1] / 16) % 2 != 0 else data.shape[1]
if (row, col) != data.shape:
    print('%s: cropping to dimensions: (%i, %i)...' % (dataset, row, col))
    data_rp = data[:row, :col]

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
segmentations, scores, subframes = [], [], []

for m in tqdm(maxima):
    position = (int(m.centroid[0] - sub_size[0] / 2),
                int(m.centroid[1] - sub_size[1] / 2),
                sub_size[0],
                sub_size[1])
    subframe = utils.get_subimg(data, position, padding='median_local')
    segmentation, score = model_is.predict(subframe[None,:,:,:])
    segmentations.append(segmentation.squeeze())
    scores.append(score[0][0])
    subframes.append(subframe[:,:,0])

# stacks images, where dim 2 is the image dimension, and 0,1 are row, col dimensions
def stack_images(imgs, offset, contrast=(0,99)):
    n = imgs.shape[2]
    stack = np.ones((imgs.shape[0]+offset*(n-1), imgs.shape[1]+offset*(n-1)))
    for i in reversed(range(n)):
        r, c = (n-i-1)*offset, i*offset
        stack[r:r+imgs.shape[0], c:c+imgs.shape[1]] = utils.enhance_contrast(imgs[:,:,i], contrast)

    return stack




## region proposal

stack = stack_images(data, 80)
img = Image.fromarray((stack*255).astype('uint8'))
img.save(os.path.join(output_folder, 'rp', 'rp_input.png'))

img = Image.fromarray((rp*255).astype('uint8'))
img.save(os.path.join(output_folder, 'rp', 'rp_output.png'))

## instance segmentation
















