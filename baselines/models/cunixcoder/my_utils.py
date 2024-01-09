from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from _utils import *

from pathlib import Path
import sys

sys.path.append(str((Path(__file__).parent.parent.parent)))
from utils.dclass import BigVulDataset
from utils import cache_dir, result_dir, get_dir
from scripts.eval_patches import BigVulDatasetPatch
import pandas as pd

logger = logging.getLogger(__name__)


def load_and_cache_defect_data_mix(args, filename, pool, tokenizer, split_tag, is_sample=False):
    dataset = 'bigvul'
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    imbalanced_df_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned_guo3.pkl'
    balanced_df_path = cache_dir() / "data/bigvul/bigvul_cleaned_guo3_balanced.pkl"
    cache_path = imbalanced_df_path

    df = pd.read_pickle(cache_path)
    if split_tag == "train":
        assert False
    elif split_tag == "valid":
        assert False
    elif split_tag == "test":
        test_df = df[df.partition == 'test']
        test_ds = BigVulDatasetPatch(test_df, dataset, partition="test", vulonly=True, sample=-1, splits="default")
        # Get mapping from index to sample ID.
        func_after = test_ds.df['func_after']
        pat = test_ds.df.copy()
        assert len(pat) == len(func_after)
        pat['func_before'] = func_after
        pat['vul'] = 0
        test_ds.df = pd.concat([test_ds.df, pat])
        # print(test_ds.df)
        test_ds.df = test_ds.df.reset_index(drop=True).reset_index()
        test_ds.df = test_ds.df.rename(columns={"index": "idx"})
        print(test_ds.stats())
        df = test_ds.df
    else:
        ValueError('wrong split_tag')

    funcs = df["func_before"].tolist()
    labels = df["vul"].tolist()
    ids = df['_id'].tolist()
    examples = []
    for i in tqdm(range(len(funcs))):
        examples.append((funcs[i], labels[i], ids[i]))
    cache_fn = ''
    if False and os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = [convert_defect_examples_to_features(tuple_example) for tuple_example in
                    tqdm(tuple_examples, total=len(tuple_examples))]
        # features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        all_ids = torch.tensor([f._id for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels, all_ids)

        # if args.local_rank in [-1, 0] and args.data_num == -1:
        #     torch.save(data, cache_fn)
    return examples, data

def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    
    # cache_fn = os.path.join(args.cache_path, split_tag)

    # examples = read_examples(filename, args.data_num, args.task)
    # if is_sample:
    #     examples = random.sample(examples, int(len(examples) * 0.1))

    # calc_stats(examples, tokenizer, is_tokenize=True)

    dataset = args.dataset # bigvul
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_minimal_cleaned_balanced.pkl'
    imbalanced_df_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned_guo3.pkl'
    balanced_df_path = cache_dir() / "data/bigvul/bigvul_cleaned_guo3_balanced.pkl"
    cache_path = balanced_df_path

    df = pd.read_pickle(cache_path)
    if split_tag == "train":
        train_df = df[df.partition == 'train'] 
        train_ds = BigVulDataset(train_df, dataset, partition="train", vulonly=False, sample=-1,
                                 splits="default", not_balance=args.not_balance) # not_balance：default ：false
        df = train_ds.df
        # file_path = args.train_data_file
    elif split_tag == "valid":
        valid_df = df[df.partition == 'valid'] 
        valid_ds = BigVulDataset(valid_df, dataset, partition="valid", vulonly=False, sample=-1,
                                 splits="default")
        df = valid_ds.df
        # file_path = args.eval_data_file
    elif split_tag == "test":
        # change testing set here 
        test_df = df[df.partition == 'test']
        test_ds = BigVulDataset(test_df, dataset, partition="test", vulonly=False, sample=-1, splits="default")
        df = test_ds.df
        # df = df[df._id== 176203] # test single function
        ## Top-25:
        # top_path = cache_dir() / "top25_df/top24.pkl" 
        # df = pd.read_pickle(top_path)    
        ## rq1 cwes set:
        # top_path = cache_dir() / "cwe_test/cwe835.pkl" 
        # df = pd.read_pickle(top_path)
    elif split_tag == "all": # save unixcoder embedding
        df = df 
        print("return all df")
    else:
        ValueError('wrong split_tag')
    
    funcs = df["func_before"].tolist() 
    labels = df["vul"].tolist() 
    ids = df['_id'].tolist() 

    examples = []
    for i in tqdm(range(len(funcs))):
        examples.append((funcs[i], labels[i], ids[i]))
    cache_fn = ''
    if False and os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample: # false
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1: # default = -1
            logger.info("Create cache data into %s", cache_fn) 
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = [convert_defect_examples_to_features(tuple_example) for tuple_example in
                    tqdm(tuple_examples, total=len(tuple_examples))]
        # features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        all_ids = torch.tensor([f._id for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels, all_ids)

        # if args.local_rank in [-1, 0] and args.data_num == -1:
        #     torch.save(data, cache_fn)
    return examples, data


def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    elif task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(data_dir, data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(data_dir, data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(data_dir, data_dir)
    elif task == 'clone':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'summarize': read_summarize_examples,
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'concode': read_concode_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
