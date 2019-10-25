## yaml test

input_data = '/home/richard/Desktop/cellfie/training_data/J123.npz'
rp_model = '/home/richard/Desktop/cellfie/models/region_proposal/191024_09.16.02/unet.347-0.149786.hdf5'
is_model = '/home/richard/Desktop/cellfie/models/instance_segmentation/191023_23.53.01/segnet.259-0.150646.hdf5'

cellfie = Cellfie(input_data, rp_model, is_model)
# segment(input_data, rp_model, is_model)