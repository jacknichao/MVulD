import pandas as pd
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent)))
from scripts.eval_patches import BigVulDatasetPatch
from utils import cache_dir
from tqdm import tqdm

if __name__ == "__main__":
    dataset = 'bigvul'
    cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    df = pd.read_pickle(cache_path)
    train_df = df[df.partition == 'train']
    valid_df = df[df.partition == 'valid']
    test_df = df[df.partition == 'test']

    train_ds = BigVulDatasetPatch(train_df, dataset, partition="train", vulonly=True, sample=-1, splits="default")
    valid_ds = BigVulDatasetPatch(valid_df, dataset, partition="valid", vulonly=True, sample=-1, splits="default")
    test_ds = BigVulDatasetPatch(test_df, dataset, partition="test", vulonly=True, sample=-1, splits="default")

    delta = train_ds.changes_stats() + valid_ds.changes_stats() + test_ds.changes_stats()
    print(delta)
    print(len(delta), sum(delta)/len(delta))
    assert False


    for i in tqdm(range(len(data.train))):
        train_func.append(data.train[i].ndata["_FVULN"].max().item())
        train_stmt += data.train[i].ndata["_VULN"].tolist()

    for i in tqdm(range(len(data.val))):
        val_func.append(data.val[i].ndata["_FVULN"].max().item())
        val_stmt += data.val[i].ndata["_VULN"].tolist()

    for i in tqdm(range(len(data.test))):
        test_func.append(data.test[i].ndata["_FVULN"].max().item())
        test_stmt += data.test[i].ndata["_VULN"].tolist()

    def funcstmt_helper(funcs, stmts):
        """Count vuln and nonvulns."""
        ret = {}
        ret["vul_funcs"] = funcs.count(1)
        ret["nonvul_funcs"] = funcs.count(0)
        ret["vul_stmts"] = stmts.count(1)
        ret["nonvul_stmts"] = stmts.count(0)
        return ret

    stats = []
    stats.append({"partition": "train", **funcstmt_helper(train_func, train_stmt)})
    stats.append({"partition": "val", **funcstmt_helper(val_func, val_stmt)})
    stats.append({"partition": "test", **funcstmt_helper(test_func, test_stmt)})

    df = pd.DataFrame.from_records(stats)
    df["func_ratio"] = df.vul_funcs / (df.vul_funcs + df.nonvul_funcs)
    df["stmt_ratio"] = df.vul_stmts / (df.vul_stmts + df.nonvul_stmts)
    df.to_csv(svd.outputs_dir() / "bigvul_stats.csv", index=0)
