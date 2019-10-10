from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, SpatialDropout2D,\
    concatenate, BatchNormalization, Flatten, Dense
from keras.optimizers import Adam


def segnet(input_size, filters=8, lr_init=.001, kernel_initializer='glorot_normal', batch_normalization=False,
           mask_weight=.5):

    inputs = Input(input_size)
    inputs_2 = BatchNormalization(input_shape=input_size)(inputs) if batch_normalization else inputs  # normalize inputs

    # 1
    conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs_2)
    conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    conv1 = BatchNormalization()(conv1) if batch_normalization else conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 1/2
    conv2 = Conv2D(filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    conv2 = BatchNormalization()(conv2) if batch_normalization else conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 1/4
    conv3 = Conv2D(filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    conv3 = BatchNormalization()(conv3) if batch_normalization else conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 1/8
    conv4 = Conv2D(filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    conv4 = BatchNormalization()(conv4) if batch_normalization else conv4
    drop4 = SpatialDropout2D(0.5)(conv4)

    # 1/4
    up7 = Conv2D(filters*4, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(drop4))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)
    conv7 = BatchNormalization()(conv7) if batch_normalization else conv7

    # 1/2
    up8 = Conv2D(filters*2, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)
    conv8 = BatchNormalization()(conv8) if batch_normalization else conv8

    # 1
    up9 = Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    conv9 = BatchNormalization()(conv9) if batch_normalization else conv9

    # predict mask
    mask = Conv2D(1, 1, activation='sigmoid', name='mask')(conv9)

    # predict class, whether there neuron is centered in subframe
    # draws from middle of network
    flat = Flatten()(conv4)
    fc1 = Dense(512, activation='relu')(flat)
    fc1 = BatchNormalization()(fc1)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = BatchNormalization()(fc2)
    is_neuron = Dense(1, activation='sigmoid', name='class')(fc2)

    # compile
    model = Model(input=inputs, output=[mask, is_neuron], name='segnet')
    losses = {'mask': 'binary_crossentropy',
              'class': 'binary_crossentropy'}
    metrics = {'mask': [],
               'class': ['accuracy']}
    loss_weights = {'mask': mask_weight, 'class': 1-mask_weight}
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=Adam(lr=lr_init), metrics=metrics)
    model.summary()

    return model
