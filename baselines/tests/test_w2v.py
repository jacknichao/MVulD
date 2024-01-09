import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_dir, processed_dir, debug, dfmp, storage_dir, data_dir, cache_dir
import utils.word2vec as w2v
import argparse


def test_w2v_bigvul_1(dataset):
    """Generate w2v embeddings on small subset of BigVul."""
    w2v.generate_w2v(dataset, sample=True)


def test_w2v_bigvul_2(dataset):
    """Load w2v embeddings."""
    path = get_dir(processed_dir() / dataset / f"w2v_False")
    myw2v = w2v.MyWord2Vec(path)
    model = myw2v.model
    # Test Most Similar
    # print(model.wv.index_to_key, model.wv.key_to_index)
    most_sim = model.wv.most_similar(['memcpy'],topn=10)
    for i in most_sim:
        print(i[0])


def test_w2v_bigvul_3(dataset):
    """Test closest embeddings."""
    path = get_dir(processed_dir() / dataset / f"w2v_True")
    myw2v = w2v.MyWord2Vec(path)

    v = myw2v.get_embeddings('hello world')
    print(v)




def test_w2v_bigvul_4(dataset):
    """Test closest embeddings."""
    path = get_dir(processed_dir() / dataset / f"w2v_True")
    myw2v = w2v.MyWord2Vec(path)

    v = myw2v.get_embeddings_list(['hello world', 'int i = a;', 'asdasdasdsa'])
    import torch
    v = torch.Tensor(v)
    print(v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    # test_w2v_bigvul_1(args.dataset)
    test_w2v_bigvul_2(args.dataset)
    # test_w2v_bigvul_3(args.dataset)
    # test_w2v_bigvul_4(args.dataset)
