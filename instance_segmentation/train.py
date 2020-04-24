from cellfie.utils import save_prediction_img
from cellfie.instance_segmentation import data_generator as dg, models
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
# import losswise
# from losswise.libs import LosswiseKerasCallback
from datetime import datetime
import os
import pickle
from glob import glob
import shutil
import yaml
import ipdb


# load configurations
with open('config.yaml', 'r') as f:
    cfg_global = yaml.safe_load(f)
with open(os.path.join('instance_segmentation', 'config.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

# if cfg['losswise_api_key']:
#     losswise.set_api_key(cfg['losswise_api_key'])  # set up losswise.com visualization
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# create model data generators
print('train datasets:')
print(cfg['train_datasets'])
print('test datasets:')
print(cfg['test_datasets'])

train_generator = dg.DataGenerator(
    cfg['train_datasets'], batch_size=cfg['batch_size'], subframe_size=cfg['subframe_size'], epoch_size=cfg['epoch_size'],
    rotation=cfg['aug_rotation'], scaling=cfg['aug_scaling'], fraction_positive_egs=cfg['fraction_positive_egs'],
    jitter=cfg['jitter'], negative_eg_distance=cfg['negative_eg_distance'], backprop_negative_masks=cfg['backprop_negative_masks'])
test_generator = dg.DataGenerator(
    cfg['test_datasets'], batch_size=cfg['batch_size'], subframe_size=cfg['subframe_size'], epoch_size=cfg['epoch_size'],
    rotation=cfg['aug_rotation'], scaling=cfg['aug_scaling'], fraction_positive_egs=cfg['fraction_positive_egs'],
    jitter=cfg['jitter'], negative_eg_distance=cfg['negative_eg_distance'], backprop_negative_masks=cfg['backprop_negative_masks'])

# create model
input_shape = (cfg['subframe_size'][0], cfg['subframe_size'][1], train_generator.shape_X[-1])
model = models.segnet(
    input_shape, filters=cfg['filters'], lr_init=cfg['lr_init'], batch_normalization=cfg['batch_normalization'],
    mask_weight=cfg['mask_weight'])


# get predictions for single batch
def save_prediction_imgs(generator, model_in, folder):
    # delete old images
    imgs_to_delete = glob(os.path.join(folder, '*.png'))[:-1]
    [os.remove(img) for img in iter(imgs_to_delete)]

    # write new images
    X, y, weights = generator[0]
    y_pred = model_in.predict(X)
    for i in range(X.shape[0]):
        file = os.path.join(folder, 'prediction%i_%.2f.png' % (i, y_pred[1][i]))
        save_prediction_img(file, X[i], y[0][i], y_pred[0][i], X_contrast=(0, 100))


# train, omg!
if cfg['use_cpu']:
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)
    K.set_session(sess)

model_folder = datetime.now().strftime('%y%m%d_%H.%M.%S')
model_path = os.path.join(cfg_global['data_dir'], 'models', 'instance_segmentation', model_folder)
os.makedirs(model_path)
callbacks = [
    ModelCheckpoint(os.path.join(model_path, '%s.{epoch:06d}-{val_loss:.6f}.hdf5' % model.name),
                    save_best_only=False, save_freq='epoch'),
    EarlyStopping(patience=cfg['early_stopping'], monitor='val_loss', verbose=1)]  # stop when validation loss stops increasing


if cfg['save_predictions_during_training']:
    callbacks.append(LambdaCallback(on_epoch_end=lambda epoch, logs: save_prediction_imgs(test_generator, model, model_path)))

# save model metadata
shutil.copyfile('config.yaml', os.path.join(model_path, 'config_global.yaml'))
shutil.copyfile(os.path.join('region_proposal', 'config.yaml'), os.path.join(model_path, 'config.yaml'))

# if cfg['losswise_api_key']:
#     callbacks.append(LosswiseKerasCallback(tag='giterdone', display_interval=1))
history = model.fit_generator(generator=train_generator, validation_data=test_generator,
                              epochs=cfg['training_epochs'], callbacks=callbacks)

with open(os.path.join(model_path, 'training_history'), 'wb') as training_file:
    pickle.dump(history.history, training_file)

# load best model and delete others
models_to_delete = glob(os.path.join(model_path, '*hdf5'))[:-1]
[os.remove(mod) for mod in iter(models_to_delete)]
model = load_model(glob(os.path.join(model_path, '*.hdf5'))[0])

# generate final predictions
save_prediction_imgs(test_generator, model, model_path)

