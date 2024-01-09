import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_dir, processed_dir, full_run_joern, dfmp, storage_dir, cache_dir, train_val_test_split_df, \
    data_dir, external_dir, debug, subprocess_cmd
import shutil
from scripts.process_dataset import cleaned_dataset, mix_patch
from process_dataset import cleaned_code
from utils.dclass import BigVulDataset
import pickle


# run txl to get all available transformations
def run_txl(filepath: str, _id, dataset, verbose=0):
    txl_path = external_dir() / "codeTransformation/RM"
    filename = external_dir() / filepath
    params = f"filename={filename}"
    script = txl_path / 'mutation.sh'
    trans = []
    resultpath = processed_dir() / f'{dataset}/aug/'
    for action in range(14):
        command = f"{str(script)} {filename}  {int(_id)} {dataset} {action}"
        # print((resultpath / f'{_id}_{action}.c').exists(), str(resultpath))
        if not (resultpath / f'{_id}_{action}.c').exists():
            print(command)
            if verbose > 2:
                debug(command)
            try:
                output = subprocess_cmd(command, verbose=verbose)
                transformed = output[0].decode()
            except Exception as e:
                print(e)
        else:
            with open(resultpath / f'{_id}_{action}.c', 'r') as f:
                transformed = f.read()
        trans.append(cleaned_code(transformed))

    source_code = trans[0]
    available_trans = []
    for action, transformed in enumerate(trans[1:]):
        if action == 0:
            continue
        if transformed != source_code:
            if action == 7:
                print(transformed.strip() != source_code.strip())
            available_trans.append(action + 1)
            # print(transformed)
    return available_trans
    # try:
    #     shutil.rmtree(external_dir() / "joern-cli_1.1.919" / "workspace" / filename.name)
    # except Exception as E:
    #     if verbose > 4:
    #         print(E)
    #     pass


def preprocess(row):
    savedir_before = get_dir(processed_dir() / row["dataset"] / "before")
    # Get C Files
    fpath1 = savedir_before / f"{row['_id']}.c"
    available_actions = run_txl(fpath1, row['_id'], dataset)
    return row['_id'], available_actions


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
    dataset_ds = BigVulDataset(df, dataset, partition="all", vulonly=False,
                               splits="default")
    df = dataset_ds.df
    print(df.shape)


    print(df.shape)

    # print(df)

    print('Data shape:', df.shape)
    print('Data columns:', df.columns)

    # splits = np.array_split(df, NUM_JOBS)
    splits = np.array_split(df, NUM_JOBS)
    # ret = dfmp(    df[df['_id'] == 995], preprocess, ordr=False, workers=16, cs=10)
    # ret = {_id:augid for _id, augid in ret}
    # with open(processed_dir() / f'{dataset}/aug/id2augid.pkl', 'wb') as f:
    #     pickle.dump(ret, f)
