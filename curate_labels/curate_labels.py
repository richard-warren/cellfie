'''
after manually curating cells with this script run curate_regions to generate json files
'''

from cells_kitchen import config as cfg
from cells_kitchen import utils
import os
import numpy as np
import sys
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd

# settings
summary_imgs_folder = r'C:\Users\erica and rick\Desktop\cells_kitchen\training_data\batch_250'
dataset = 'nf.01.00'
channels = ['corr', 'median', 'std']
contrast = [1, 99.9]
crop = 100
scaling = 4
fig1_size = 6
fig2_size = 3


# load summary images
summary_imgs = np.load(os.path.join(summary_imgs_folder, dataset + '.npz'), allow_pickle=True)
X = [utils.enhance_contrast(summary_imgs['X'][()][k], contrast) for k in channels]
height, width = X[0].shape[0], X[0].shape[1]

# compute borders
borders = utils.get_targets(os.path.join(cfg.data_dir, 'labels', dataset),
                            border_thickness=1, use_curated_labels=False)['borders']
cell_num = borders.shape[0]
cell = 0
main_img = utils.add_contours(X[0], borders.max(0), color=(0,0,1))

# create or load previous data
file = os.path.join(cfg.data_dir, 'labels', dataset, 'cells_to_include.csv')
if os.path.exists(file):
    data = pd.read_csv(file)
else:
    # create new spreadsheet
    data = pd.DataFrame({'cell': np.arange(0, cell_num), 'include': np.zeros(cell_num)})
    data.to_csv(file, index=False)
    print('new spreadsheet created')


# handle keypress events
def keypress(event):

    sys.stdout.flush()
    global cell

    if event.key == 'right':
        if cell == cell_num - 1:
            cell = 0
        else:
            cell += 1

    elif event.key == 'left':
        if cell == 0:
            cell = cell_num - 1
        else:
            cell -= 1

    elif event.key == 'enter':
        data.loc[cell, 'include'] = 0 if data.loc[cell, 'include'] else 1
        data.to_csv(file, index=False)

    elif event.key == 'escape':
        plt.close(fig1)
        plt.close(fig2)

    show_cell()


# set up figures
plt.close('all')
plt.rcParams['toolbar'] = 'None'  # disable toolbars

fig1 = plt.figure(figsize=(fig1_size, fig1_size*(height/width)), facecolor='black')
ax1 = plt.axes(position=[0, 0, 1, 1])
im1 = ax1.imshow(np.ones((height, width, 3)), cmap='bone')
fig1.canvas.mpl_connect('key_press_event', keypress)

fig2 = plt.figure(figsize=(fig2_size*len(channels), fig2_size), facecolor='black')
ax2 = plt.axes(position=[0, 0, 1, 1])
im2 = ax2.imshow(np.ones((crop, crop*len(channels), 3)), cmap='bone')
fig2.canvas.mpl_connect('key_press_event', keypress)


def show_cell():

    print('showing %s, cell %i/%i' % (dataset, cell, cell_num))
    color = [0, 1, 0] if data.include[cell] else [1, 0, 0]

    # find crop position
    center = np.argwhere(borders[cell]).mean(axis=0).astype('int')
    ylims = slice(max(center[0] - crop // 2, 0), min(center[0] + crop // 2, height))
    xlims = slice(max(center[1] - crop // 2, 0), min(center[1] + crop // 2, width))

    # get and show subframe
    img = [utils.add_contours(x[ylims, xlims], borders[cell][ylims, xlims], color=color) for x in X]
    img = np.concatenate(img, axis=1)
    img = resize(img, (img.shape[0]*scaling, img.shape[1]*scaling, 3), mode='constant')
    im2.set_data(img)
    fig2.canvas.draw()

    # highlight subframe in fig1
    mask = np.ones((height, width, 3)) * .2
    mask[ylims, xlims] = 1
    im1.set_data(utils.add_contours(main_img, borders[cell], color=color) * mask)
    fig1.canvas.draw()


show_cell()




