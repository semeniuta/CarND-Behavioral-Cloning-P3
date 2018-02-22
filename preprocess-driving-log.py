import os
import pandas as pd
import argparse


def open_original(fname):

    colnames = ['center','left','right','steering','throttle','brake','speed']
    df =  pd.read_csv(fname, names=colnames)
    return df


def preprocess_in_df(df):

    for i in df.index:
        for col in ('center', 'left', 'right'):
            orig = df.loc[i][col]
            pth = orig.split('IMG')[-1]
            new_val = 'IMG' + pth
            df.at[i, col] = new_val

def save(df, fname):
    df.to_csv(fname, index=False)


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description='Preprocess driving log.')
    arg_parser.add_argument('--src', required=True, type=str)
    arg_parser.add_argument('--dst', required=True, type=str)
    args = arg_parser.parse_args()

    df = open_original(args.src)
    preprocess_in_df(df)
    save(df, args.dst)
