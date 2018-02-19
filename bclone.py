import os
import numpy as np
import pandas as pd
import cv2


def load_log(data_dir):

    logfile = os.path.join(data_dir, 'driving_log.csv')
    return pd.read_csv(logfile)


def load_data(
    data_dir,
    im_groups=('center', 'left', 'right'),
    controls=('steering', 'throttle', 'speed', 'brake'),
    indices=None
):

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
