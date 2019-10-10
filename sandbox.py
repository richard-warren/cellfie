## write sample training images

import utils
utils.write_sample_imgs(X_contrast=(0,99))
utils.write_sample_border_imgs(channels=['corr', 'median'], contrast=(1, 99))
# write_sample_border_imgs(channels=['corr'], contrast=(0,99))

## try auto-correlation images

dataset = 'J123'

folder = os.path.join(cfg.data_dir, 'caiman', 'datasets', 'images_'+dataset)
imgs = utils.get_frames(folder, frame_inds=np.arange(0, 1000))

##
offset = 1
acorr = np.sum(np.multiply(imgs[:-offset], imgs[offset:]), axis=0)
acorr = acorr / (np.sqrt(np.sum(np.square(imgs[:-offset]), 0)) * np.sqrt(np.sum(np.square(imgs[offset:]), 0)))
# acorr = np.sum(np.multiply(imgs[:-offset], imgs[offset:]), 0)
# acorr = np.divide(acorr, np.square(np.linalg.norm(imgs, axis=0)))

# plt.close('all')
# imshow = plt.imshow(utils.scale_img(acorr))
imshow.set_data(utils.scale_img(acorr))