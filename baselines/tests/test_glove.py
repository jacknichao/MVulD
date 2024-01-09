import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_dir, processed_dir, debug, dfmp, storage_dir, data_dir, cache_dir
import utils.glove as glove
import argparse

def test_glove_bigvul_1(dataset):
    """Generate glove embeddings on small subset of BigVul."""
    glove.generate_glove(dataset, sample=True)


def test_glove_bigvul_2(dataset):
    """Load glove embeddings."""
    path = processed_dir() / f"{dataset}/glove_False/vectors.txt"
    _, corp = glove.glove_dict(path, cache=False)
    print(list(corp.keys())[:10])
    assert corp["if"] <= 10
    # assert corp["ps"] <= 10
    assert corp["return"] <= 10
    # assert corp["len"] <= 10


def test_glove_bigvul_3(dataset):
    """Test closest embeddings."""
    path = processed_dir() / f"{dataset}/glove_False/vectors.txt"
    gdict, _ = glove.glove_dict(path, cache=False)
    if_closest = glove.find_closest_embeddings("if", gdict)
    print(if_closest)
    assert if_closest[0] == "if"
    assert if_closest[1] == "else"


def test_glove_bigvul_4(dataset):
    """Test get embeddings."""
    path = glove.processed_dir() / f"{dataset}/glove_False/vectors.txt"
    gdict, _ = glove.glove_dict(path, cache=False)
    ret = glove.get_embeddings("if outofvocabword", gdict, 100)
    assert any([i != 0.01 for i in ret[0]])
    assert all([i == 0.001 for i in ret[1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    # test_glove_bigvul_1(args.dataset)
    test_glove_bigvul_2(args.dataset)
    test_glove_bigvul_3(args.dataset)
    test_glove_bigvul_4(args.dataset)
