import dgl
import sys
from dataset import BigVulDatasetDevign
from model import DevignModel
from pathlib import Path
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str((Path(__file__).parent.parent.parent)))
from utils import debug, get_run_id, processed_dir, get_metrics_probs_bce, cache_dir, set_seed, result_dir, get_dir
from utils.dclass import BigVulDataset
from utils.my_log import LogWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import argparse
import pandas as pd


def evaluate(model, val_dl, val_ds, logger, args):
    model.eval()
    with torch.no_grad():
        all_probs = torch.empty((0)).float().to(args.device)
        all_logits = torch.empty((0)).float().to(args.device)
        all_true = torch.empty((0)).float().to(args.device)
        for val_batch in tqdm(val_dl, total=len(val_dl), desc='Validing...'):
            val_batch = val_batch.to(args.device)
            val_labels = dgl.max_nodes(val_batch, "_LABEL")
            val_probs, val_logits = model(val_batch, val_ds)
            # val_preds = logits.argmax(dim=1)
            all_probs = torch.cat([all_probs, val_probs])
            all_logits = torch.cat([all_logits, val_logits])
            all_true = torch.cat([all_true, val_labels])
        val_mets = get_metrics_probs_bce(all_true, all_probs, all_logits)
    return val_mets


def test(model, test_dl, test_ds, logger, args):
    # logger.load_best_model()
    dataset2id = {
        'reveal': '202207081438_v1',
        'bigvul':'202208221701_v1', # balanced best-f1
        # 'bigvul':'202208222128_v1', # imbalanced best-f1
        'devign': '202207081440_v1'
    }
    balanced_path = result_dir() / f"devign/{args.dataset}" / f"{dataset2id[args.dataset]}/best_f1.model"
    imbalanced_path = result_dir() / f"devign/{args.dataset}/imbalanced"/ f"{dataset2id[args.dataset]}/best_f1.model"
    path = balanced_path

    model.load_state_dict(torch.load(path))
    model.eval()
    all_probs = torch.empty((0)).float().to(args.device)
    all_logits = torch.empty((0)).float().to(args.device)
    all_true = torch.empty((0)).float().to(args.device)
    all_ids = torch.empty((0)).float().to(args.device)
    with torch.no_grad():
        for test_batch in test_dl:
            test_batch = test_batch.to(args.device)
            test_labels = dgl.max_nodes(test_batch, "_LABEL")
            test_ids = dgl.max_nodes(test_batch, "_SAMPLE")

            test_probs, test_logits = model(test_batch, test_ds)
            
            all_probs = torch.cat([all_probs, test_probs])
            all_logits = torch.cat([all_logits, test_logits])
            all_true = torch.cat([all_true, test_labels])
            all_ids = torch.cat([all_ids, test_ids])

        test_mets = get_metrics_probs_bce(all_true, all_probs, all_logits)
    if logger:
        logger.test(test_mets)
    else:
        print(test_mets)
        # for i in zip(all_ids, all_true, all_pred):
        #     if i[1] == 1:
        #         print(i)
    return test_mets


def train(model, train_dl, train_ds, val_dl, val_ds, test_dl, test_ds, logger, args):
    # %% Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    loss_function = nn.BCELoss(reduction='sum')
    for epoch in range(args.epochs):
        for batch in tqdm(train_dl, total=len(train_dl), desc='Training...'):
            # Training
            model.train()
            batch = batch.to(args.device)
            probs, logits = model(batch, train_ds)
            labels = dgl.max_nodes(batch, "_LABEL")
            # print('labels:', labels.shape)
            # print('logits:', logits.shape)
            # print(logits, labels)
            loss = loss_function(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
            train_mets = get_metrics_probs_bce(labels, probs, logits)
            # Evaluation
            # pred = logits.argmax(dim=1).cpu().detach().numpy()
            # print('pred:', pred.shape)
            val_mets = None
            if logger.log_val():
                val_mets = evaluate(model, val_dl=val_dl, val_ds=val_ds, logger=logger, args=args)
            logger.log(train_mets, val_mets)
            logger.save_logger()

        # Early Stopping
        if logger.stop():
            break
        logger.epoch()

    # Print test results
    test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    configs = json.load(open('./config.json'))
    for item in configs:
        args.__dict__[item] = configs[item]
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 指定了gpu0？
    # print(f"device={args.device}")

    # %% Load data
    dataset = args.dataset
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    balanced_df_path = cache_dir() / "data/bigvul/bigvul_cleaned3_balanced.pkl"
    # imbalanced_df_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned3.pkl'
    cache_path = balanced_df_path

    df = pd.read_pickle(cache_path)
    print(df["partition"].value_counts()) #
    train_df = df[df.partition == 'train']
    valid_df = df[df.partition == 'valid']
    test_df = df[df.partition == 'test']

    dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}

    test_ds = BigVulDatasetDevign(df=test_df, partition="test", dataset=dataset, vulonly=False)
    test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)
    # print(test_ds.df[test_ds.df.vul == 1]._id.values)
    # args.max_patience = args.val_every * args.max_patience
    # %% Create model
    dev = args.device
    model = DevignModel(input_dim=args.input_size, output_dim=args.hidden_size)
    model.to(dev)

    set_seed(args)
    # Train loop
    # logger.load_logger()
    if args.do_train:
        train_ds = BigVulDatasetDevign(df=train_df, partition="train", dataset=dataset, not_balance=args.not_balance)
        val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset=dataset)
        
        dl_args = {"drop_last": False, "shuffle": True, "num_workers": 6}
        train_dl = GraphDataLoader(train_ds, batch_size=args.train_batch_size, **dl_args)
        
        dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
        val_dl = GraphDataLoader(val_ds, batch_size=args.test_batch_size, **dl_args)
        
        args.val_every = int(len(train_dl))
        args.log_every = int(len(train_dl) / 5)
        
        # %% Create Logger
        ID = get_run_id(args={})
        # ID = "202108121558_79d3273"
        logger = LogWriter(
            model, args, path=get_dir(result_dir() / f"devign/{args.dataset}"/ ID)
        )
        debug(args)
        logger.info(args)

        train(model, train_dl=train_dl, train_ds=train_ds, val_dl=val_dl, val_ds=val_ds,
              test_dl=test_dl, test_ds=test_ds, logger=logger, args=args)
        test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=logger)
    if args.do_test:
        test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=False)


if __name__ == '__main__':
    main()
    # dataset = 'reveal'
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    # df = pd.read_pickle(cache_path)
    # train_df = df[df.partition == 'train']
    # valid_df = df[df.partition == 'valid']
    # test_df = df[df.partition == 'test']
    # val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset=dataset)
    #
    # val_ds.item(10)
