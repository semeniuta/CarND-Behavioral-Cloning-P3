import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import argparse

tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

import bclone # <- the core collection of helper functions


DATASETS = {
    'STD': 'bclone-data-standard',
    'MYDRIVE_1': 'my_2018-03-13',
    'MYDRIVE_2':'my_2018-03-18-1',
    'MYDRIVE_3': 'my_2018-03-18-2',
    'MYDRIVE_4': 'my_2018-03-27', # a big set of my own data
    'SPECIAL': 'my_2018-03-27-special', # set of special situations
    'SPECIAL2': 'my_2018-03-27-special2' # more special situations
}


def nn_model(prob=0.5, dropout_for_dense=True):

    model = Sequential()

    model.add( Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)) )
    model.add( Cropping2D(cropping=((70, 25), (0, 0))) )

    model.add( Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu') )
    model.add(Dropout(prob))
    model.add( Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu') )
    model.add(Dropout(prob))
    model.add( Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu') )
    model.add(Dropout(prob))
    model.add( Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu') )
    model.add(Dropout(prob))

    model.add( Flatten() )

    model.add( Dense(100, activation='relu') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(50, activation='relu') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(10, activation='relu') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(1) )

    model.compile(loss='mse', optimizer='adam')

    return model


def load_data_custom(data_dir):
    '''
    Custom data preparation procedure. Loads all datasets
    except for 'SPECIAL2', and then appends the first 1500 entries
    of 'SPECIAL2' to the resulting dataframe
    '''

    # Load the following dataset entirely

    datasets_full = (
        'STD',
        'MYDRIVE_1',
        'MYDRIVE_2',
        'MYDRIVE_3',
        'MYDRIVE_4',
        'SPECIAL'
    )

    data_paths = [os.path.join(data_dir, DATASETS[key]) for key in datasets_full]

    log_df = bclone.load_and_combine_logs(*data_paths)

    # Add a part of 'SPECIAL2' dataset

    special2 = os.path.join(data_dir, DATASETS['SPECIAL2'])
    df_special2 = bclone.load_log(special2)
    bclone.add_data_dir_info_to_df(df_special2, special2)

    log_df = bclone.combine_dataframes(
        log_df,
        df_special2[:1500]
    )

    return log_df


def train_model(log_df, save_dir):

    model = nn_model(prob=0.4, dropout_for_dense=False)

    filepath = os.path.join(save_dir, 'weights-improvement-{epoch:02d}-{val_loss:.4f}.h5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    train_df, valid_df, history = bclone.train(
        model,
        log_df,
        batch_sz=50,
        epochs=20,
        left_correction=0.3,
        right_correction=-0.3,
        callbacks=callbacks_list
    )

    return model, history, train_df, valid_df


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='Train a neural network model for cloning driving behavior.')
    arg_parser.add_argument('--datadir', default='..')
    arg_parser.add_argument('--savedir', default='..')
    args = arg_parser.parse_args()

    print('Loading the data')
    log_df = load_data_custom(args.datadir)

    print('Training the model')
    model, history, train, valid = train_model(log_df, args.savedir)
