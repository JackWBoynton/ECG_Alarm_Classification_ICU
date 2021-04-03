import random as rd
import datetime
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import sys
import sklearn.model_selection

THRESHOLD_up = 0.9*2500
THRESHOLD_down = 0.1*2500

def resnet_att(in_shape: tuple, classes: int, n_feature_maps: int = 64, inc=2500, X_shape = (1, 2500)) -> [keras.Model]:
    """ ResNet with an attention layer at the output for temporal addition."""

    input_layer = keras.layers.Input(batch_input_shape=(250, *X_shape),dtype='float32')

    ## BEGIN RESNET ##
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    output_block_3 = keras.layers.Dropout(0.4)(output_block_3)
    ## END RESNET ##

    den = keras.layers.Bidirectional(keras.layers.LSTM(n_feature_maps, stateful=True, return_sequences=True))(output_block_3)
    den = keras.layers.Dropout(0.1)(den)
    den = keras.layers.BatchNormalization()(den)
    inter = keras.layers.Dense(100, activation='relu')(den)
    den = keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu'))(inter)
    gap = keras.layers.GlobalAveragePooling1D()(den)
    inter = keras.layers.Dense(100, activation='relu')(gap)

    output_layer_real = keras.layers.Dense(1, activation='sigmoid')(inter)
    conc = keras.layers.Concatenate(axis=-1)([tf.expand_dims(output_layer_real, 1), input_layer])
    model = keras.models.Model(inputs=[input_layer], outputs=[conc,output_layer_real])
    print(model.summary())
    return model


import time


def time_loss():
    @tf.function
    def loss(y_true, y_pred):
        pred = y_pred[:, :, 0]
        dat = y_pred[:, :, 1:]

        arr = tf.TensorArray(tf.float32, size=y_pred.shape[0])
        for i in range(pred.shape[0]):
            x = pred[i]
            x_dat = dat[i]
            res = K.min(tf.where(x_dat != 0))
            if res.shape != (1,1):
                if y_true[i] == 1 and x > THRESHOLD_up:
                    arr = arr.write(i, K.cast(res, tf.float32))
                elif y_true[i] == 0 and x < THRESHOLD_down:
                    arr = arr.write(i, K.cast(res, tf.float32))
            else:
                arr = arr.write(i, 1.043)
        return tf.reduce_sum(arr.stack())
    return loss

def run(incc, X, Y, x, y):
    global THRESHOLD_up
    global THRESHOLD_down
    THRESHOLD_up = 0.9*X.shape[2]
    THRESHOLD_down = 0.1*X.shape[2]
    mod = resnet_att((250,), 1, inc=incc, X_shape=X.shape[1:])

    mod.compile(keras.optimizers.SGD(0.0010), loss=[time_loss(), keras.losses.BinaryCrossentropy()], metrics={'dense_3': 'accuracy'}, loss_weights=(1, 1))

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0000001)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=20, restore_best_weights=True)
    mod.fit(x, y, validation_data=(X,Y), epochs=100, batch_size=250, callbacks=[reduce_lr, es])
    mod.save("mod.hdf5")
    return mod
