import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import cache_dir, external_dir,dfmp, get_dir, processed_dir, subprocess_cmd, project_dir
from utils.dclass import BigVulDataset
import pandas as pd


def run_joern(filepath: str, verbose: int):
    """Extract graph using most recent Joern."""
    """
                      int myfunc(int b) {
                        int a = 42;
                        if (b > 10) {
                           foo(a);
                        }
                        bar(a);
                      }
                      """

    script_file = external_dir() / "get_func_graph.sc"
    filename = external_dir() / filepath
    params = f"filename={filename}"
    command = str(external_dir() / "joern-cli_1.1.919" / command)
    if verbose > 2:
        debug(command)
    subprocess_cmd(command, verbose=verbose)
    try:
        shutil.rmtree(external_dir() / "joern-cli_1.1.919" / "workspace" / filename.name)
    except Exception as E:
        if verbose > 4:
            print(E)
        pass


def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = get_dir(processed_dir() / row["dataset"] / "before")
    savedir_after = get_dir(processed_dir() / row["dataset"] / "after")
    # print(savedir_after, savedir_before)
    # Write C Files
    fpath1 = savedir_before / f"{row['_id']}.c"

    txl_path = project_dir() / 'codeTransformation'
    command = f"{str(txl_path)}/runner.sh {fpath1}  {str(txl_path)} 8"
    print(command)
    output = subprocess_cmd(command)

    return output


if __name__ == '__main__':
    # main()
    dataset = 'bigvul_mix'
    cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    df = pd.read_pickle(cache_path)
    train_df = df[df.partition == 'train']
    valid_df = df[df.partition == 'valid']
    test_df = df[df.partition == 'test']
    test_ds = BigVulDataset(df=test_df, partition="test", dataset=dataset, sample=2)

    output = dfmp(test_ds.df, preprocess, ordr=True, workers=8)

    print(output[1][0])
    print(output[1][1])
    # val_ds.cache_features()

    # val_ds = BigVulDatasetIVDetect(df=valid_df, partition="valid", dataset=dataset)
    # val_ds.item(10)
