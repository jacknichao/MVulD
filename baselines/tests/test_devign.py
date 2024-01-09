import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.devign.dataset import BigVulDatasetDevign, count_labels
from models.ivdetect.dataset import BigVulDatasetIVDetect
from utils import cache_dir
import pandas as pd

if __name__ == '__main__':
    # main()
    dataset = 'reveal'
    cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    df = pd.read_pickle(cache_path)
    train_df = df[df.partition == 'train']
    valid_df = df[df.partition == 'valid']
    test_df = df[df.partition == 'test']
    val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset=dataset)
    val_ds[10]

    # val_ds.cache_features()

    # val_ds = BigVulDatasetIVDetect(df=valid_df, partition="valid", dataset=dataset)
    # val_ds.item(10)
