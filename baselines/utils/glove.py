"""Python wrapper to call StanfordNLP Glove."""

import pickle as pkl
from pathlib import Path

import numpy as np
from scipy import spatial
from utils import get_dir, processed_dir, cache_dir, hashstr, project_dir, watch_subprocess_cmd, dfmp, tokenize_lines, \
    tokenize
import pandas as pd


def glove(
        CORPUS,
        VOCAB_FILE="vocab.txt",
        COOCCURRENCE_FILE="cooccurrence.bin",
        COOCCURRENCE_SHUF_FILE="cooccurrence.shuf.bin",
        SAVE_FILE="vectors",
        VERBOSE=2,
        MEMORY=4.0,
        VOCAB_MIN_COUNT=5,
        VECTOR_SIZE=200,
        MAX_ITER=15,
        WINDOW_SIZE=15,
        BINARY=2,
        NUM_THREADS=8,
        X_MAX=10,
):
    """Run StanfordNLP Glove on a corpus. Mainly copied from demo.sh in their repo."""
    savedir = Path(CORPUS).parent
    VOCAB_FILE = savedir / VOCAB_FILE
    COOCCURRENCE_FILE = savedir / COOCCURRENCE_FILE
    COOCCURRENCE_SHUF_FILE = savedir / COOCCURRENCE_SHUF_FILE
    SAVE_FILE = savedir / SAVE_FILE
    BUILDDIR = project_dir() / 'GloVe/build' # glove存放的位置：默认在该项目下面 myproject/GloVe

    cmd1 = f"{BUILDDIR}/vocab_count \
        -min-count {VOCAB_MIN_COUNT} \
        -verbose {VERBOSE} \
        < {CORPUS} > {VOCAB_FILE}"

    cmd2 = f"{BUILDDIR}/cooccur \
        -memory {MEMORY} \
        -vocab-file {VOCAB_FILE} \
        -verbose {VERBOSE} \
        -window-size {WINDOW_SIZE} \
        < {CORPUS} > {COOCCURRENCE_FILE}"

    cmd3 = f"{BUILDDIR}/shuffle \
        -memory {MEMORY} \
        -verbose {VERBOSE} \
        < {COOCCURRENCE_FILE} > {COOCCURRENCE_SHUF_FILE}"

    cmd4 = f"{BUILDDIR}/glove \
        -save-file {SAVE_FILE} \
        -threads {NUM_THREADS} \
        -input-file {COOCCURRENCE_SHUF_FILE} \
        -x-max {X_MAX} -iter {MAX_ITER} \
        -vector-size {VECTOR_SIZE} \
        -binary {BINARY} \
        -vocab-file {VOCAB_FILE} \
        -verbose {VERBOSE}"

    watch_subprocess_cmd(cmd1)
    watch_subprocess_cmd(cmd2)
    watch_subprocess_cmd(cmd3)
    watch_subprocess_cmd(cmd4)


def glove_dict(vectors_path, cache=True):
    """Load glove embeddings and vocab.

    Example:
    vectors_path = svd.processed_dir() / "bigvul/glove/vectors.txt"
    """
    # Caching
    savepath = get_dir(cache_dir() / "glove")
    savepath /= str(hashstr(str(vectors_path)))
    if cache:
        try:
            with open(savepath, "rb") as f:
                return pkl.load(f)
        except Exception as E:
            print(E)
            pass

    # Read into dict
    embeddings_dict = {}
    with open(vectors_path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    # Read vocab
    with open(vectors_path.parent / "vocab.txt", "r") as f:
        vocab = [i.split()[0] for i in f.readlines()]
        vocab = dict([(j, i) for i, j in enumerate(vocab)])

    # Cache
    with open(savepath, "wb") as f:
        pkl.dump([embeddings_dict, vocab], f)

    return embeddings_dict, vocab


def find_closest_embeddings(word, embeddings_dict, topn=10):
    """Return closest GloVe embeddings."""
    embedding = embeddings_dict[word]
    return sorted(
        embeddings_dict.keys(),
        key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding),
    )[:topn]


def get_embeddings(text: str, emb_dict: dict, emb_size: int = 100) -> np.array:
    """Get embeddings from text, zero vectors for OoV.

    Args:
        text (str): Text as string. Should be preprocessed, tokenised by space.
        emb_dict (dict): Dict with key = token and val = embedding.
        emb_size (int, optional): Size of embedding. Defaults to 100.

    Returns:
        np.array: Array of embeddings, shape (seq_length, emb_size)
    """
    return [
        emb_dict[i] if i in emb_dict else np.full(emb_size, 0.001) for i in text.split()
    ]


def get_embeddings_list(li: list, emb_dict: dict, emb_size: int = 100) -> list:
    """Get embeddings from a list of sentences, then average.

    Args:
        li (list): List of sentences.
        emb_dict (dict): Dict with key = token and val = embedding.
        emb_size (int, optional): Size of embedding. Defaults to 100.

    Example:
    li = ['static long ec device ioctl xcmd struct cros ec dev ec void user arg',
        'struct cros ec dev ec',
        'void user arg',
        'static long',
        'struct cros ec dev ec',
        'void user arg',
        '']

    glove_path = svd.processed_dir() / "bigvul/glove/vectors.txt"
    emb_dict, _ = glove_dict(glove_path)
    emb_size = 200
    """
    li = [tokenize(i) for i in li]
    li = [i if len(i) > 0 else "<EMPTY>" for i in li]
    return [np.mean(get_embeddings(i, emb_dict, emb_size), axis=0) for i in li]


def generate_glove(dataset="bigvul", sample=False, cache=True, n=None):
    """Generate Glove embeddings for tokenised dataset."""
    # if os.path.exists(savedir / "vectors.txt") and cache:
    #     debug("Already trained GloVe.")
    #     return
    # if dataset == "bigvul":
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl' # kw
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_minimal_cleaned_balanced.pkl'
    balanced_df_path = cache_dir() / "data/bigvul/bigvul_cleaned_guo3_balanced.pkl"
    cache_path = balanced_df_path
    df = pd.read_pickle(cache_path)
    if sample:
        df = df.sample(n if n else len(df), random_state=0)
    MAX_ITER = 15 if sample else 500

    # Only train GloVe embeddings on train samples
    samples = df[df.partition == "train"].copy()
    # samples = df[df.label == "train"].copy()

    # Preprocessing
    print('sample shape:', samples.shape)
    samples.func_before = dfmp(
        samples, tokenize_lines, "func_before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.func_before.to_numpy() for i in j]  # get corpus
    # samples.before = dfmp(
    #     samples, tokenize_lines, "before", cs=200, desc="Get lines: "
    # )
    # lines = [i for j in samples.before.to_numpy() for i in j]  # get corpus

    # Save corpus
    savedir = get_dir(processed_dir() / dataset / f"glove_{sample}")
    with open(savedir / "corpus.txt", "w") as f:
        f.write("\n".join(lines))

    # Train Glove Model
    CORPUS = savedir / "corpus.txt"
    glove(CORPUS, MAX_ITER=MAX_ITER)
