import logging

from utils import tokenize, processed_dir, get_dir, debug, cache_dir, dfmp, tokenize_lines
from gensim.models import Word2Vec
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np


def generate_w2v(dataset="bigvul", sample=False, cache=True, embedding_size=100,
                 window_size=10, n=None):
    """Train Word2Vec model for tokenised dataset."""
    savedir = get_dir(processed_dir() / dataset / f"w2v_{sample}")

    if os.path.exists(savedir / "w2v.model") and cache:
        debug("Already trained Word2Vec.")
        return
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl' 
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_minimal_cleaned_balanced.pkl'
    balanced_df_path = cache_dir() / "data/bigvul/bigvul_cleaned3_balanced.pkl"
    cache_path = balanced_df_path
    
    df = pd.read_pickle(cache_path)
    if sample:
        df = df.sample(n if n else int(len(df) * 0.3), random_state=0)
    MAX_ITER = 15 if sample else 500

    # Only train Word2Vec on train samples
    samples = df[df.partition == "train"].copy()
    # samples = df[df.label == "train"].copy()

    # Preprocessing
    # samples.before = dfmp(
    #     samples, tokenize_lines, "before", cs=200, desc="Get lines: ", workers=32
    # )
    # lines = [' '.join(j).split() for j in samples.before.to_numpy()]  # get corpus
    samples.func_before = dfmp(
        samples, tokenize_lines, "func_before", cs=200, desc="Get lines: ", workers=32
    )
    lines = [' '.join(j).split() for j in samples.func_before.to_numpy()]  # get corpus
    
    
    # Train Word2Vec model
    model = train_w2v(lines, embedding_size=100, epochs=MAX_ITER,
                      window_size=10)
    model.save(str(savedir / "w2v.model"))


def train_w2v(sentences, embedding_size=100, epochs=5,
              window_size=10):
    print('Total Examples:', len(sentences))
    wvmodel = Word2Vec(sentences, workers=16, size=embedding_size, window=window_size)
    print('Embedding Size : ', wvmodel.vector_size) 

    for _ in tqdm(range(epochs), total=epochs):
        wvmodel.train(sentences, total_examples=len(sentences), epochs=1)

    return wvmodel


def load_w2v(path: str):
    """Load Word2Vec model.

    path = svd.processed_dir() / "bigvul/w2v_False"
    """
    path = str(path)
    if path.split("/")[-1] != "w2v.model":
        path += "/w2v.model"
    return Word2Vec.load(path)


class MyWord2Vec:
    """Word2Vec class."""

    def __init__(self, path: str):
        """Init class."""
        self.model = load_w2v(path)

    def get_embeddings(self, sentence: str, emb_size: int = 100) -> list:
        """Get embeddings from the words of sentence, then average.

        Args:
            li (list): List of sentences.
            emb_size (int, optional): Size of embedding. Defaults to 100.
        """
        tokens = tokenize(sentence).split()
        # a = np.mean([
        #     self.model.wv[i] if i in self.model.wv else np.zeros(100) for i in tokens
        # ], axis=0)
        # print(a.dtype)
        if len(tokens):
            return np.mean([
                self.model.wv[i] if i in self.model.wv else np.zeros(100) for i in tokens
            ], axis=0).astype(np.float32)
        return np.zeros(100)

    def get_embeddings_list(self, li: list, emb_size: int = 100) -> list:
        """Get embeddings from a list of sentences

        Args:
            li (list): List of sentences.
            emb_size (int, optional): Size of embedding. Defaults to 100.
        """
        return [self.get_embeddings(i, emb_size) for i in li]
