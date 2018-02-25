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
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

import bclone


DATA_DIR_STD = '../bclone-data-standard'
DATA_DIR_NEW = '../my_2018-02-22'


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


def get_startegy(id):

    func_name = 'strategy_{:d}'.format(id)
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

    log_df = bclone.load_and_combine_logs(DATA_DIR_STD, DATA_DIR_NEW)

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
    log_df_new = bclone.load_and_combine_logs(DATA_DIR_NEW)

    model = nvidia_model()
    train_dfs, valid_dfs, history = bclone.train_multiple_sets(
        model,
        [log_df_std, log_df_new],
        [40, 8],
        epochs=2
    )

    return model, history, train_dfs, valid_dfs



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='Train a neural network model for clonign driving behavior.')
    arg_parser.add_argument('--strategy', type=int, default=3)
    args = arg_parser.parse_args()

    strategy = get_startegy(args.strategy)
    model, history, train, valid = strategy()
