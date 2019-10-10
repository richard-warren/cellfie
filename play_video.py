from cells_kitchen import config as cfg
from cells_kitchen import utils
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import tifffile
from tqdm import tqdm


# settings
dataset = 'nf.01.00'
frames_to_show = 2000
fps = 20
close_when_done = True
frames_to_sample = 100
percentiles = [1, 99.5]
add_borders = True


folder = os.path.join(cfg.data_dir, 'datasets', 'images_'+dataset)

# find images
files = glob.glob(os.path.join(folder, '*.tif*'))
frames_to_show = min(frames_to_show, len(files))

# get pixel percentiles using random sample of images
inds = np.random.choice(frames_to_show, min(frames_to_sample, frames_to_sample))
imgs = tifffile.imread(np.array(files)[inds].tolist()).astype('float32')
lims = np.percentile(imgs, percentiles)

if add_borders:
    borders = utils.get_targets(
        os.path.join(cfg.data_dir, 'labels', dataset), border_thickness=1, collapse_masks=True)['borders']


def close_window(event):
    if event.key == 'escape':
        global escaping
        escaping = True
        plt.close()


## initialize window
plt.close('al')
fig = plt.figure(num=folder, figsize=(6.4, 6.4*(imgs.shape[1]/imgs.shape[2])))
fig.canvas.mpl_connect('key_press_event', close_window)
im = plt.imshow(np.zeros((1, 1, 3), dtype='float32'))
ax = plt.gca()
ax.set_position([0, 0, 1, 1])
plt.show()

escaping = False

for f in tqdm(files[0:frames_to_show]):
    frame = tifffile.imread(f).astype('float32')
    frame = utils.enhance_contrast(frame, percentiles)
    if add_borders:
        frame = utils.add_contours(frame, borders, color=(1, 0, 0))
    im.set_data(frame)
    plt.pause(1/fps)
    if escaping:
        break

if close_when_done:
    plt.close()
