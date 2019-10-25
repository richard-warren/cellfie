from cellfie.end_to_end.Cellfie import Cellfie
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np


# settings
rp_model = '/home/richard/Desktop/cellfie/models/region_proposal/191024_09.16.02/unet.347-0.149786.hdf5'
is_model = '/home/richard/Desktop/cellfie/models/instance_segmentation/191023_23.53.01/segnet.259-0.150646.hdf5'

# initializations
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cellfie = Cellfie(rp_model, is_model)


##
for d in cfg['datasets']:
    print('%s: analyzing session' % d)
    cellfie.analyze_data(os.path.join(cfg['data_dir'], 'training_data', d+'.npz'))
    plt.savefig(os.path.join(cfg['data_dir'], 'results', 'figures', d+'_masks.png'))
    np.savez(os.path.join(cfg['data_dir'], 'results', 'masks', d+'_masks'),
             masks=cellfie.cell_masks, scores=cellfie.scores)
plt.close('all')
