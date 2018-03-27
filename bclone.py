import os
import numpy as np
import pandas as pd
import sklearn
import cv2
from sklearn.model_selection import train_test_split


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
    '''
    Infinite data generator for use in Keras' model.fit_generator
    that loads data from multiple sets at once. Each data frame in
    log_dataframes has the associated batch size (in batch_sizes),
    which internally form the data_generator per set. Different
    batch_sizes control how much data samples are drawn from the corresponding
    data sets. On every invocation of the generator,
    the yielded data subsets are combined.
    '''

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


def fit_gen(model, train_gen, valid_gen, log_df_train, log_df_valid, n_epochs, **fit_kwargs):
    '''
    Helper function for calling Keras' model.fit_generator.
    Returns the resulting history object.
    '''

    history = model.fit_generator(
        train_gen,
        samples_per_epoch=len(log_df_train)*3,
        validation_data=valid_gen,
        nb_val_samples=len(log_df_valid)*3,
        nb_epoch=n_epochs,
        **fit_kwargs
    )

    return history


def train(model, log_df, batch_sz, epochs, **fit_kwargs):
    '''
    Train the given model with the available set of data
    '''

    train, valid = train_test_split(log_df, test_size=0.2)
    valid_gen = data_generator(valid, batch_size=batch_sz, controls=['steering'])
    train_gen = data_generator(train, batch_size=batch_sz, controls=['steering'])

    history = fit_gen(model, train_gen, valid_gen, train, valid, n_epochs=epochs, **fit_kwargs)

    return train, valid, history


def determine_n_samples(dfs, batch_sizes):

    sizes = [len(df) for df in dfs]
    smallest_size_idx = np.argmin(sizes)

    factor = len(dfs[smallest_size_idx]) // batch_sizes[smallest_size_idx]

    return 3 * factor * sum(batch_sizes)


def train_multiple_sets(
    model,
    log_dataframes,
    batch_sizes,
    epochs,
    valid_share=0.2,
    **fit_kwargs
):
    '''
    Train the given model with multiple sets of data using
    data_generator_from_muiltiple_sets.

    valid_share -- percentage of the validation set size compared to the
                   total number of samples
    fit_kwargs -- keyworded arguments forwarded to the model.fit_generator
                  function
    '''

    splitted = [train_test_split(df, test_size=valid_share) for df in log_dataframes]
    train_dfs = [t for t, v in splitted]
    valid_dfs = [v for t, v in splitted]

    valid_gen = data_generator_from_muiltiple_sets(valid_dfs, batch_sizes, controls=['steering'])
    train_gen = data_generator_from_muiltiple_sets(train_dfs, batch_sizes, controls=['steering'])

    n_samples_train = determine_n_samples(train_dfs, batch_sizes)
    n_samples_valid = determine_n_samples(valid_dfs, batch_sizes)

    history = model.fit_generator(
        train_gen,
        samples_per_epoch=n_samples_train,
        validation_data=valid_gen,
        nb_val_samples=n_samples_valid,
        nb_epoch=epochs,
        **fit_kwargs
    )

    return train_dfs, valid_dfs, history
