import pickle
import pandas as pd
from utils import cache_dir
from utils.dclass import BigVulDataset

dataset ='bigvul'
cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'

df = pd.read_pickle(cache_path)
split_tag = 'train'
if split_tag == "train":
    train_df = df[df.partition == 'train']
    train_ds = BigVulDataset(train_df, dataset, partition="train", vulonly=False, sample=-1,
                             splits="default")
    df = train_ds.df
    # file_path = args.train_data_file
elif split_tag == "valid":
    valid_df = df[df.partition == 'valid']
    valid_ds = BigVulDataset(valid_df, dataset, partition="valid", vulonly=False, sample=-1,
                             splits="default")
    df = valid_ds.df
    # file_path = args.eval_data_file
elif split_tag == "test":
    test_df = df[df.partition == 'test']
    test_ds = BigVulDataset(test_df, dataset, partition="test", vulonly=False, sample=-1, splits="default")
    df = test_ds.df
df = df['func_before']
print(df)
df.to_csv('../simCSE/')