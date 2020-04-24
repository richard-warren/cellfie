from cellfie.end_to_end.Cellfie import Cellfie
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np


# settings
rp_model = r'E:\cellfie_data\models\region_proposal\200424_13.42.45\unet.000027-0.408449.hdf5'
is_model = r'E:\cellfie_data\models\instance_segmentation\200424_13.50.17\segnet.001000-0.187542.hdf5'
maxima_thresh = .1  # (0->1) in the region proposal network output, only maxima above maxima_thresh will be passed along to the instance segmentation network
min_distance = 4  # (pixels) only include maxima that are greater than min_distance pixels from one another


with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cellfie = Cellfie(rp_model, is_model)

for d in cfg['datasets']:
    print('%s: analyzing session' % d)
    cellfie.analyze_data(os.path.join(cfg['data_dir'], 'training_data', d+'.npz'), maxima_thresh=maxima_thresh, min_distance=min_distance)
    plt.savefig(os.path.join(cfg['data_dir'], 'results', 'figures', d+'_masks.png'))
    print('saving: %s' % os.path.join(cfg['data_dir'], 'results', 'masks', d+'_masks'))
    np.savez(os.path.join(cfg['data_dir'], 'results', 'masks', d+'_masks'), masks=cellfie.cell_masks, scores=cellfie.scores)
plt.close('all')
