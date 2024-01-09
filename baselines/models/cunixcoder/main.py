# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
import time

from model import DefectModel
from configs import add_args, set_seed
from my_utils import get_filenames, get_elapse_time, load_and_cache_defect_data, load_and_cache_defect_data_mix
from model import get_model_size

from pathlib import Path
import sys

sys.path.append(str((Path(__file__).parent.parent.parent)))
from utils.dclass import BigVulDataset
from utils import cache_dir, result_dir, get_dir
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,average_precision_score
from unixcoder import UniXcoder

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 # 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 # 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = 16

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        id = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label) # logit在model里已经经过softmax了

            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            ids.append(id.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    ids = np.concatenate(ids, 0)
    preds = logits[:, 1] > 0.5
    eval_acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"logits.shape={logits.shape}")
    print(logits[:, 1])
    pr_auc = average_precision_score(labels, logits[:, 1]) # (true,probs)
    result = {
        "eval_acc": float(eval_acc),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_prauc": float(pr_auc),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    easy_to_copy = f"{round(eval_acc, 4)} / {round(precision, 4)} / {round(recall, 4)} / {round(f1, 4)}"
    logger.info("%s", easy_to_copy)
    # if write_to_pred:
    #     with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
    #         for example, pred in zip(eval_examples, preds):
    #             if pred:
    #                 f.write(str(example._id) + '\t1\n')
    #             else:
    #                 f.write(str(example._id) + '\t0\n')
    if args.do_patch:
        # cache_path = get_dir(result_dir() / f"patched_result/{args.dataset}/unixcoder/no_balance_{args.not_balance}")
        cache_path = get_dir(result_dir() / f"patched_result/{args.dataset}/unixcoder/")
        pickle.dump(
            (ids, labels, logits, preds),
            open(str(cache_path / f'result_True'), 'wb'))
        get_representation(model, dataloader=eval_dataloader, args=args, cache_path=cache_path, store=True)
    else:
        result_df = pd.DataFrame(zip(ids, labels, preds), columns=['y_ids', 'y_trues', 'y_preds'])
        output_dir = os.path.join(args.output_dir, '{}'.format("result.csv"))
        result_df.to_csv(output_dir)
    return result

def get_representation(model, dataloader, cache_path, args, store=False):

    logger.info("***** Running unixcoder get representation *****")

    model.eval() 

    balanced_path = result_dir() / f"unixcoder/{args.dataset}/checkpoint-best-f1/pytorch_model.bin"
    # imbalanced_path = result_dir() / f"unixcoder/{args.dataset}/not_balance/checkpoint-best-f1/pytorch_model.bin"
    path = balanced_path
    model.load_state_dict(torch.load(path)) 

    if not isinstance(dataloader, DataLoader):
        sampler = SequentialSampler(dataloader)
        dataloader = DataLoader(dataloader, sampler=sampler, batch_size=args.eval_batch_size)
    all_repr = []
    all_labels = []
    all_ids = []
    for batch in tqdm(dataloader, total=len(dataloader), desc="Getting Representation"):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        id = batch[2]
        with torch.no_grad():
            _repr, label = model.get_repr(inputs, label)
            all_repr.append(_repr.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            if id.shape[0] == 2 * label.shape[0]:
                id = id[::2]
            all_ids.append(id)
    all_repr = np.concatenate(all_repr, 0) 
    all_labels = np.concatenate(all_labels, 0)
    all_ids = np.concatenate(all_ids, 0)
    if store:
        pickle.dump(
            (all_ids, all_repr, all_labels),
            open(str(cache_path/f'embeddings_True'), 'wb'))
        result_df = pd.DataFrame(zip(all_ids, all_repr, all_labels), columns=['ids', 'repr', 'lable'])
        output_dir = os.path.join(cache_path, '{}'.format("result.csv"))
        pkl_output_dir = cache_path /'result.pkl'
        print(f"output_dir={output_dir}")
        print(f"pkl_output_dir={pkl_output_dir}")
        result_df.to_csv(output_dir) 
        result_df.to_pickle(pkl_output_dir)
    else:
        result_df = pd.DataFrame(zip(all_ids, all_repr, all_labels), columns=['ids', 'repr', 'lable'])
        pkl_output_dir = cache_path /'result.pkl'
        print(f"pkl_output_dir={pkl_output_dir}")
        result_df.to_pickle(pkl_output_dir) 
    return all_ids, all_repr, all_labels


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    # output_dir
    args.output_dir = str(result_dir() / f'unixcoder_4/{args.dataset}') 

    if args.not_balance:
        args.output_dir = os.path.join(args.output_dir, 'not_balance') 
    args.summary_dir = str(f'{args.output_dir}/summary') 
    set_seed(args)

    # Build model
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    unixcoder = UniXcoder(args.model_name_or_path) # default="microsoft/unixcoder-base-nine"
    config = unixcoder.config
    model = unixcoder.model
    tokenizer = unixcoder.tokenize

    model = DefectModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None: # 
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    # pool = multiprocessing.Pool(cpu_cont)
    pool = None
    # args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    # save_unix_embed = False # 
    if args.save_unixcoder_embedding:
    # if save_unix_embed:
        logger.info("***** save_unixcoder_embedding *****")
        ## load all data:save all the embedding from unixcoder
        all_data_examples, all_data = load_and_cache_defect_data(args, args.train_filename, pool, tokenizer, 'all',
                                                                is_sample=False)
        all_sampler = SequentialSampler(all_data) #
        all_dataloader = DataLoader(all_data, sampler=all_sampler, batch_size=args.eval_batch_size)
        logger.info("  Num examples = %d", len(all_data))
        logger.info("  Num batches = %d", len(all_dataloader))
        logger.info("  Batch size = %d", args.eval_batch_size)
        ## cache path
        cache_path = get_dir(cache_dir() / f"unixcoder_output/{args.dataset}/")
        # str(result_dir() / f'unixcoder/{args.dataset}')
        if args.not_balance:
           cache_path = get_dir(cache_dir()/ f"unixcoder_output/{args.dataset}/not_balance/")
        get_representation(model, dataloader=all_dataloader, args=args, cache_path=cache_path, store=False)
        return 

    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(args.summary_dir)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_defect_data(args, args.train_filename, pool, tokenizer, 'train',
                                                                is_sample=False)

        eval_examples, eval_data = load_and_cache_defect_data(args, args.dev_filename, pool, tokenizer,
                                                              'valid', is_sample=False)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)
        max_steps = args.num_train_epochs * len(train_dataloader)
        # evaluate the model per epoch
        save_steps = len(train_dataloader)
        args.warmup_steps = max_steps // 5

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_f1 = 0, 0
        not_f1_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, labels, _ = batch

                loss, logits = model(source_ids, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0 and args.do_eval:
                    ## validate set
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    result = evaluate(args, model, eval_examples, eval_data)
                    eval_f1 = result['eval_f1']

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_f1', round(eval_f1, 4), cur_epoch)

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    if True or args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_f1 > best_f1:
                        not_f1_inc_cnt = 0
                        logger.info("  Best f1: %s", round(eval_f1, 4))
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best f1 changed into %.4f\n" % (cur_epoch, round(eval_f1, 4)))
                        best_f1 = eval_f1
                        # Save best checkpoint for best f1
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best f1 model into %s", output_model_file)
                    else:
                        not_f1_inc_cnt += 1
                        logger.info("f1 does not increase for %d epochs", not_f1_inc_cnt)
                        if not_f1_inc_cnt > args.patience:
                            logger.info("Early stop as f1 do not increase for %d times", not_f1_inc_cnt)
                            fa.write("[%d] Early stop as not_f1_inc_cnt=%d\n" % (cur_epoch, not_f1_inc_cnt))
                            is_early_stop = True
                            break

                model.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

    if args.do_test or args.do_patch:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-f1']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file)) 

            if args.n_gpu > 1:
                # multi-gpu training
                model = torch.nn.DataParallel(model)
            if args.do_patch:
                eval_examples, eval_data = load_and_cache_defect_data_mix(args, args.test_filename, pool, tokenizer,
                                                                          'test',False)
            else:
                eval_examples, eval_data = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test',
                                                                      False)
              
            result = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
            logger.info("  test_f1=%.4f", result['eval_f1'])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-f1: %.4f\n" % (criteria, result['eval_f1']))
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] f1: %.4f\n\n" % (
                        criteria, result['eval_f1']))
    fa.close()


if __name__ == "__main__":
    main()
