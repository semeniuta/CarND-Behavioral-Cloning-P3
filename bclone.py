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


def add_data_dir_info_to_df(df, data_dir):
    '''
    Modify in-place the supplied driving log dataframe so that
    image paths are concatenated with the corresponding
    data_dir
    '''

    for i in df.index:
        for col in ('center', 'left', 'right'):
            orig = df.loc[i][col]
            new_val = os.path.join(data_dir, orig.strip())
            df.at[i, col] = new_val


def load_and_combine_logs(*dirs):
    '''
    Load driving logs from the supplied directories
    as a single Pandas dataframe
    '''

    dataframes = []
    for d in dirs:
        df = load_log(d)
        add_data_dir_info_to_df(df, d)
        dataframes.append(df)

    res = dataframes[0]
    for df in dataframes[1:]:
        res = res.append(df, ignore_index=True)

    return res


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


def open_images(log_df, controls):
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

            imfile = line[c]
            im = cv2.imread(imfile)

            im_list.append(im)
            controls_list.append(controls_vals)

    X = np.array(im_list)
    y = np.array(controls_list)

    if y.shape[1] == 1:
        y = y.reshape(-1)

    return X, y


def data_generator(log_df, batch_size=32, controls=['steering', 'throttle', 'speed', 'brake']):
    '''
    Infinite data generator for use in Keras' model.fit_generator.
    Opens images from all three cameras using open_images.
    batch_size is counted in terms of the driving log entries,
    so the yielded batches will be 3 times larger
    '''

    log_df = sklearn.utils.shuffle(log_df)
    n = len(log_df)

    while True:
        for offset in range(0, n, batch_size):
            log_subset = log_df[offset:offset+batch_size]
            X_batch, y_batch = open_images(log_subset, controls)
            yield sklearn.utils.shuffle(X_batch, y_batch)


def data_generator_from_muiltiple_sets(log_dataframes, batch_sizes, controls=['steering', 'throttle', 'speed', 'brake']):

    assert len(log_dataframes) == len(batch_sizes)

    generators = [data_generator(df, sz, controls) for df, sz in zip(log_dataframes, batch_sizes)]

    while True:

        x = []
        y = []

        for gen in generators:
            X_batch, y_batch = next(gen)
            x.append(X_batch)
            y.append(y_batch)

        yield sklearn.utils.shuffle(np.vstack(x), np.hstack(y))


def fit_gen(model, train_gen, valid_gen, log_df_train, log_df_valid, n_epochs):

    history = model.fit_generator(
        train_gen,
        samples_per_epoch=len(log_df_train)*3,
        validation_data=valid_gen,
        nb_val_samples=len(log_df_valid)*3,
        nb_epoch=n_epochs
    )

    return history
