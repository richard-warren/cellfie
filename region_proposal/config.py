# network
X_layers = ['corr', 'median', 'std']  # summary images to include as input to the network // ['corr', 'mean', 'median', 'max', 'std']
y_layers = ['somas']  # summary images to include as target for the network // ['somas', 'borders', 'centroids']
high_pass_sigma = 15  # std of gaussian based high pass filtering of inptus // set to False to turn off high pass filtering
filters = 16  # numbers of filters in first layer of network // scales up as network deepens // seemed to work with as few as 16 // 8 was a little blurrier

# training
# datasets = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST', 'K53', 'J115', 'J123', 'nf.01.00', 'nf.02.01', 'nf.04.01']

# fair split
# train_datasets = ['N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'K53', 'J115', 'J123', 'nf.01.00', 'nf.02.01']
# test_datasets = ['N.00.00', 'YST', 'nf.04.01']

# equal test and train
train_datasets = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST', 'K53', 'J115', 'J123', 'nf.01.00', 'nf.02.01', 'nf.04.01']
test_datasets = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST', 'K53', 'J115', 'J123', 'nf.01.00', 'nf.02.01', 'nf.04.01']

# accidental split that may have worked
# test_datasets = ['N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'K53', 'J115', 'J123', 'nf.01.00', 'nf.02.01']
# train_datasets = ['N.00.00', 'YST', 'nf.04.01']

subframe_size = (160, 160)  # each dimension must be divisible by four
use_cpu = False  # whether to use CPU instead of GPU for training
aug_rotation = True  # whether to apply 0, 90, 180, or 270 degree rotations randomly
aug_scaling = (.75, 1.25)  # min and max image scaling // set to (1, 1) for no scaling
batch_normalization = True
losswise_api_key = '3ZGMSXASM'  # set to False if not using losswise.com
lr_init = .001
batch_size = 16
epoch_size = 4  # number of batches in an epoch
training_epochs = 5000  # epochs
early_stopping = 200  # epochs
save_predictions_during_training = True  # set whether to save images of predictions at each epoch end during training
