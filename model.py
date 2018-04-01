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

import bclone


DATA_DIR_STD = '../bclone-data-standard'
DATA_DIR_FAULTS = '../my_2018-02-22'
DATA_DIR_MYDRIVE_1 = '../my_2018-03-13'
DATA_DIR_MYDRIVE_2 = '../my_2018-03-18-1'
DATA_DIR_MYDRIVE_3 = '../my_2018-03-18-2'
DATA_DIR_MYDRIVE_4 = '../my_2018-03-27' # a big set of my own data
DATA_DIR_SPECIAL = '../my_2018-03-27-special' # another set of special situations
DATA_DIR_SPECIAL2 = '../my_2018-03-27-special2' # red stuff and shadows


def nvidia_model():

    model = Sequential()

    model.add( Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)) )
    model.add( Cropping2D(cropping=((70, 25), (0, 0))) )

    model.add( Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu') )
    model.add( Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu') )
    model.add( Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu') )

    model.add( Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu') )
    #model.add( Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu') )

    model.add( Flatten() )

    model.add( Dense(100) )
    model.add( Dense(50) )
    model.add( Dense(10) )
    model.add( Dense(1) )

    model.compile(loss='mse', optimizer='adam')

    return model


def nvidia_model_2(prob=0.5, dropout_for_dense=True):

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


def nvidia_model_3(prob=0.5, dropout_for_dense=True):

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

    model.add( Dense(100, activation='sigmoid') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(50, activation='sigmoid') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(10, activation='sigmoid') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(1) )

    model.compile(loss='mse', optimizer='adam')

    return model


def nvidia_model_4(prob=0.5, dropout_for_dense=True):

    model = Sequential()

    model.add( Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)) )
    model.add( Cropping2D(cropping=((70, 25), (0, 0))) )

    model.add( Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu') )
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

    model.add( Dense(25, activation='relu') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(10, activation='relu') )
    if dropout_for_dense:
        model.add(Dropout(prob))

    model.add( Dense(1) )

    model.compile(loss='mse', optimizer='adam')

    return model


def mymodel(prob=0.5, dropout_for_dense=False):

    model = Sequential()

    model.add( Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 3)) )
    model.add( Cropping2D(cropping=((70, 25), (0, 0))) )

    model.add( Convolution2D(24, 5, 5, subsample=(1, 1), activation='relu') )
    #model.add( MaxPooling2D(pool_size=(1, 2)) )
    model.add( Dropout(prob) )

    model.add( Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu') )
    #model.add( MaxPooling2D(pool_size=(1, 2)) )
    model.add( Dropout(prob) )

    model.add( Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu') )
    model.add( Dropout(prob) )

    model.add( Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu') )
    model.add( Dropout(prob) )

    model.add( Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu') )
    model.add( Dropout(prob) )

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


def get_startegy(id):

    func_name = 'strategy_{}'.format(id)
    return getattr(sys.modules[__name__], func_name)


def strategy_1():

    log_df = bclone.load_and_combine_logs(DATA_DIR_STD)

    model = nvidia_model()

    train_df, valid_df, history = bclone.train(
        model,
        log_df,
        batch_sz=10,
        epochs=2
    )

    return model, history, train_df, valid_df


def strategy_2():

    log_df = bclone.load_and_combine_logs(DATA_DIR_STD, DATA_DIR_FAULTS)

    model = nvidia_model()

    train_df, valid_df, history = bclone.train(
        model,
        log_df,
        batch_sz=100,
        epochs=2
    )

    return model, history, train_df, valid_df


def strategy_3():

    log_df_std = bclone.load_and_combine_logs(DATA_DIR_STD)
    log_df_new = bclone.load_and_combine_logs(DATA_DIR_FAULTS)

    model = nvidia_model()
    train_dfs, valid_dfs, history = bclone.train_multiple_sets(
        model,
        [log_df_std, log_df_new],
        [40, 8],
        epochs=2
    )

    return model, history, train_dfs, valid_dfs


def strategy_multi():

    log_df_std = bclone.load_and_combine_logs(DATA_DIR_STD)
    log_df_faults = bclone.load_and_combine_logs(DATA_DIR_FAULTS)
    log_df_mydrive_1 = bclone.load_and_combine_logs(DATA_DIR_MYDRIVE_1)
    log_df_mydrive_2 = bclone.load_and_combine_logs(DATA_DIR_MYDRIVE_2)
    log_df_mydrive_3 = bclone.load_and_combine_logs(DATA_DIR_MYDRIVE_3)

    datasets = [log_df_std, log_df_mydrive_1, log_df_mydrive_2, log_df_mydrive_3, log_df_faults]

    for df in datasets:
        print(len(df))

    # This was one pretty good
    #model = nvidia_model_2(prob=0.2, dropout_for_dense=False)
    model = mymodel(prob=0.2, dropout_for_dense=False)

    filepath="weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    train_dfs, valid_dfs, history = bclone.train_multiple_sets(
        model,
        datasets,
        [28, 7, 7, 7, 12],
        epochs=20,
        valid_share=0.2,
        callbacks=callbacks_list
    )

    return model, history, train_dfs, valid_dfs


def strategy_kiss():

    log_df = bclone.load_and_combine_logs(
        DATA_DIR_STD,
        DATA_DIR_MYDRIVE_1,
        DATA_DIR_MYDRIVE_2,
        DATA_DIR_MYDRIVE_3,
        DATA_DIR_MYDRIVE_4,
        DATA_DIR_SPECIAL,
    )

    df_special2 = bclone.load_log(DATA_DIR_SPECIAL2)
    bclone.add_data_dir_info_to_df(df_special2, DATA_DIR_SPECIAL2)

    log_df = bclone.combine_dataframes(
        log_df,
        df_special2[:1500] # a hack so not to take the entire dataset
    )

    #model = mymodel(prob=0.4, dropout_for_dense=True)
    model = nvidia_model_2(prob=0.4, dropout_for_dense=False)

    filepath="weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"
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

    arg_parser = argparse.ArgumentParser(description='Train a neural network model for clonign driving behavior.')
    arg_parser.add_argument('--strategy', default='3')
    args = arg_parser.parse_args()

    strategy = get_startegy(args.strategy)
    model, history, train, valid = strategy()
