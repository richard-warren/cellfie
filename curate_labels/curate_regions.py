'''
after choosing which cells to include (curate_labels.py), run this script to create new json file that include only the cells that were marked for inclusion
'''

from cells_kitchen import config as cfg
import os
import pandas as pd
import numpy as np
import json

root_dir = os.path.join(cfg.data_dir, 'labels')
datasets = next(os.walk(root_dir))[1]  # all datasets in labels folder

for d in datasets:

    # trying loading spreadsheet enconding which cells to include
    include_file = os.path.join(root_dir, d, 'cells_to_include.csv')

    if os.path.exists(include_file):

        # read original cells and restrict to cells to be included
        with open(os.path.join(root_dir, d, 'regions', 'consensus_regions.json')) as jfile:
            include_cell = np.array(pd.read_csv(include_file).include, dtype='bool')
            regions = json.load(jfile)
            regions_sub = [c for (c, f) in zip(regions, include_cell) if f]

        # write new json file of cells to be included
        with open(os.path.join(root_dir, d, 'regions', 'consensus_regions_curated.json'), 'w') as jfile:
            json.dump(regions_sub, jfile)
            print('writing file: %s', os.path.join(root_dir, d, 'regions', 'consensus_regions_curated.json'))

