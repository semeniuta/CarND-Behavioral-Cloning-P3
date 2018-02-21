import os
import numpy as np
import pandas as pd
import sklearn
import cv2


def load_log(data_dir):
    '''
    Load driving log as a Pandas dataframe
    '''

    logfile = os.path.join(data_dir, 'driving_log.csv')
    return pd.read_csv(logfile)


def load_data(
    data_dir,
    im_groups=('center', 'left', 'right'),
    controls=('steering', 'throttle', 'speed', 'brake'),
    indices=None
):
    '''
    Load all specified data into memory at once
    '''

    log_df = load_log(data_dir)

    if indices is None:
        idx_from, idx_to = 0, len(log_df)
        data_range = range(len(log_df))
    else:
        idx_from, idx_to = indices
        data_range = range(*indices)

    controls_df = log_df[list(controls)].iloc[idx_from:idx_to]

    images = {}
    for grp in im_groups:
        lst = []
        for i in data_range:
            imfile = os.path.join(data_dir, log_df.iloc[i][grp])
            im = cv2.imread(imfile)
            lst.append(im)
        images[grp] = np.array(lst)

    return images, controls_df


def open_images(data_dir, log_df, controls):
    '''
    For a given driving log data frame (or a subset of one),
    open images from all three cameras and gather them with
    the corresponding measuremenets from the log
    '''

    cameras = ('center', 'left', 'right')

    im_list = []
    controls_list = []
    for i in range(len(log_df)):
        line = log_df.iloc[i]
        controls_vals = line[controls]
        for c in cameras:

            imfile = os.path.join(data_dir, line[c].strip())
            im = cv2.imread(imfile)

            im_list.append(im)
            controls_list.append(controls_vals)

    X = np.array(im_list)
    y = np.array(controls_list)

    if y.shape[1] == 1:
        y = y.reshape(-1)

    return X, y


def data_generator(data_dir, batch_size=32, controls=['steering', 'throttle', 'speed', 'brake']):
    '''
    Infinite data generator for use in Keras' model.fit_generator.
    Opens images from all three cameras using open_images
    '''

    log_df = sklearn.utils.shuffle( load_log(data_dir) )
    n = len(controls)

    while True:
        for offset in range(0, n, batch_size):
            log_subset = log_df[offset:offset+batch_size]
            X_batch, y_batch = open_images(data_dir, log_subset, controls)
            yield sklearn.utils.shuffle(X_batch, y_batch)
