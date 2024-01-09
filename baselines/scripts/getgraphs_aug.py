import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
sys.path.append(str(Path(__file__).parent.parent))

from utils.dclass import BigVulDataset
from utils import get_dir, processed_dir, full_run_joern, dfmp, storage_dir, cache_dir, train_val_test_split_df, \
    data_dir
from scripts.process_dataset import cleaned_dataset, mix_patch
from glob import glob


def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = get_dir(processed_dir() / row["dataset"] / "aug")
    # print(savedir_after, savedir_before)
    # Write C Files
    finished = glob(str(processed_dir() / f"{dataset}/aug/{row['_id']}_*.c"))
    for i in finished:
        fpath1 = i
        # Run Joern o  n "before" code
        if not os.path.exists(f"{fpath1}.edges.json"):
            full_run_joern(fpath1, verbose=3)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    # parser.add_argument('--split', required=True, type=int, default=0)

    args = parser.parse_args()
    print(args)
    dataset = args.dataset

    # SETUP
    NUM_JOBS = 1000
    # JOB_ARRAY_NUMBER = 0 if "ipykernel" in sys.argv[0] else int(sys.argv[1]) - 1

    # Read Data
    cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    df = pd.read_pickle(cache_path)
    for part in ['train', 'valid', 'test']:
        dataset_ds = BigVulDataset(df, dataset, partition=part, vulonly=False,
                                   splits="default")
        df = dataset_ds.df
        print(df.shape)

        # print(df)

        print('Data shape:', df.shape)
        print('Data columns:', df.columns)

        # splits = np.array_split(df, NUM_JOBS)
        splits = np.array_split(df, NUM_JOBS)
        dfmp(df, preprocess, ordr=False, workers=16)
