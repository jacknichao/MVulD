import dgl
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent)))
from models.ivdetect.dataset import feature_extraction as ivdetect_feature_extraction
from models.devign.dataset import feature_extraction as devign_feature_extraction
from models.devign.dataset import type_map, type_one_hot
from models.ivdetect.model import IVDetect
from models.devign.model import DevignModel
from models.reveal.ggnn.model import GGNNSum
from models.reveal.model import MetricLearningModel
from models.reveal.graph_dataset import DataSet
from models.reveal.trainer import predict, evaluate_patch
from tqdm import tqdm
import json
import warnings
import pickle
from utils.word2vec import MyWord2Vec

warnings.filterwarnings('ignore')

from utils import debug, get_run_id, processed_dir, get_metrics_logits, cache_dir, set_seed, result_dir, get_dir, dfmp, \
    get_metrics_probs_bce
import utils.glove as glove
import utils.word2vec as word2vec
from utils.my_log import LogWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import argparse
import pandas as pd
import numpy as np
from glob import glob


class BigVulDatasetPatch:
    """Represent BigVul as graph dataset."""
    DATASET = None

    def __init__(self, df, dataset, partition="train", vulonly=True, sample=-1, splits="default"):
        """Init class."""
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(processed_dir() / f"{dataset}/before/*nodes*"))
        ]
        BigVulDatasetPatch.DATASET = dataset
        self.df = df

        # Filter only vulnerable
        if vulonly:
            self.df = self.df[self.df.vul == 1]
        self.partition = partition
        self.df = self.df[self.df.partition == partition]
        self.df = self.df[self.df._id.isin(self.finished)]

        # filter large functions
        print(f'{partition} before large(VUL):', len(self.df))
        self.df = self.df[self.df['func_before'].map(lambda x: len(x.splitlines())) <= 500]
        print(f'{partition} after large(VUL):', len(self.df))

        # filter large functions
        print(f'{partition} before large(PATCH):', len(self.df))
        self.df = self.df[self.df['func_after'].map(lambda x: len(x.splitlines())) <= 500]
        print(f'{partition} after large(PATCH):', len(self.df))

        # Balance training set
        # if partition == "train":
        #     vul = self.df[self.df.vul == 1]
        #     nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
        #     self.df = pd.concat([vul, nonvul])

        # Filter out samples with no lineNumber from Joern output
        self.df["valid"] = dfmp(
            self.df, BigVulDatasetPatch.check_validity, "_id", desc="Validate Samples: ", workers=32,
        )
        print(f'{partition} before valid:', len(self.df))
        self.df = self.df[self.df.valid]
        print(f'{partition} after valid:', len(self.df))
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)
        # Add delta columns
        self.df['delta'] = self.changes_stats()

    def itempath_vul(_id):
        """Get itempath path from item id."""
        return processed_dir() / f"{BigVulDatasetPatch.DATASET}/before/{_id}.c"

    def itempath_pat(_id):
        """Get itempath path from item id."""
        return processed_dir() / f"{BigVulDatasetPatch.DATASET}/after/{_id}.c"

    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        try:
            with open(str(BigVulDatasetPatch.itempath_vul(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False
            with open(str(BigVulDatasetPatch.itempath_vul(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
            # with open(str(BigVulDatasetPatch.itempath_pat(_id)) + ".nodes.json", "r") as f:
            #     nodes = json.load(f)
            #     lineNums = set()
            #     for n in nodes:
            #         if "lineNumber" in n.keys():
            #             lineNums.add(n["lineNumber"])
            #             if len(lineNums) > 1:
            #                 valid = 1
            #                 break
            #     if valid == 0:
            #         return False
            # with open(str(BigVulDatasetPatch.itempath_pat(_id)) + ".edges.json", "r") as f:
            #     edges = json.load(f)
            #     edge_set = set([i[2] for i in edges])
            #     if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
            #         return False
            return True
        except Exception as E:
            print(E, str(BigVulDatasetPatch.itempath_pat(_id)))
            return False

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["partition", "vul"]).count()[["idx"]])

    def changes_stats(self):
        func_before = self.df.func_before.map(lambda x: len(x.splitlines())).tolist()
        func_after = self.df.func_after.map(lambda x: len(x.splitlines())).tolist()
        import math
        delta = [math.fabs((func_after[i] - func_before[i])) for i in range(len(func_before))]
        return delta

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"BigVulDatasetPatch(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"


class BigVulDatasetIVDetectPatch(BigVulDatasetPatch):
    """IVDetect version of BigVul."""

    def __init__(self, **kwargs):
        """Init."""
        super(BigVulDatasetIVDetectPatch, self).__init__(**kwargs)
        # Load Glove vectors.
        glove_path = processed_dir() / f"{kwargs['dataset']}/glove_False/vectors.txt"
        self.emb_dict, _ = glove.glove_dict(glove_path)
        # self.df = self.df[self.df._id == 174374]
        # filter large functions
        print(f'{kwargs["partition"]} LOCAL before large:', len(self.df))
        ret = dfmp(
            self.df,
            BigVulDatasetIVDetectPatch._feat_ext_itempath,
            "_id",
            ordr=True,
            desc="Cache features: ",
            workers=32
        )
        self.df = self.df[ret]
        print(f'{kwargs["partition"]} LOCAL after large:', len(self.df))

        # Get mapping from index to sample ID.
        print(self.df.columns)

        func_after = self.df['func_after']
        pat = self.df.copy()
        assert len(pat) == len(func_after)
        pat['func_before'] = func_after
        pat['vul'] = 0
        self.df = pd.concat([self.df, pat])
        # print(self.df)
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        print('stats:', self.stats())
        # print(self.df.columns, self.df.shape)
        # print(self.df.vul)
        self.idx2id = pd.Series(self.df._id.values, index=self.df.idx).to_dict()

    def item(self, _id, is_eval=False):
        # print('item', _id, is_eval)
        """Get item data."""
        if is_eval:
            n, _ = ivdetect_feature_extraction(BigVulDatasetPatch.itempath_pat(_id))
        else:
            n, _ = ivdetect_feature_extraction(BigVulDatasetPatch.itempath_vul(_id))
        n.subseq = n.subseq.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))
        n.nametypes = n.nametypes.apply(
            lambda x: glove.get_embeddings(x, self.emb_dict, 200)
        )
        n.data = n.data.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))
        n.control = n.control = n.control.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))

        asts = []

        def ast_dgl(row, lineid):
            if len(row) == 0:
                return None
            '''
            row example
            [[0, 0, 0, 0, 0, 0], 
             [1, 2, 3, 4, 5, 6], 
             ['int alloc addbyter int output FILE data', 'int output', 'FILE data', '', 'int', 'int output', 'FILE data']]

            '''
            outnode, innode, ndata = row
            g = dgl.graph((outnode, innode))
            g.ndata["_FEAT"] = torch.Tensor(
                np.array(glove.get_embeddings_list(ndata, self.emb_dict, 200))
            )
            g.ndata["_ID"] = torch.Tensor([_id] * g.number_of_nodes())
            g.ndata["_LINE"] = torch.Tensor([lineid] * g.number_of_nodes())
            return g

        for row in n.itertuples():
            asts.append(ast_dgl(row.ast, row.id))

        return {"df": n, "asts": asts}

    def _feat_ext_itempath(_id):
        """Run feature extraction with itempath."""
        n, e = ivdetect_feature_extraction(BigVulDatasetPatch.itempath_vul(_id))
        n1, e1 = ivdetect_feature_extraction(BigVulDatasetPatch.itempath_pat(_id))
        return len(n) > 0 and len(n) <= 500 and len(n1) > 0 and len(n1) <= 500

    def get_vul_label(self, idx):
        """Obtain vulnerable or not."""
        return idx < (len(self.df) / 2)

    def __getitem__(self, idx):
        """Override getitem."""
        _id = self.idx2id[idx]
        if idx < len(self.df) / 2:
            n, e = ivdetect_feature_extraction(BigVulDatasetPatch.itempath_vul(_id))
        else:
            n, e = ivdetect_feature_extraction(BigVulDatasetPatch.itempath_pat(_id))
        # n["vuln"] = n.id.map(self.get_vuln_indices(_id)).fillna(0)

        g = dgl.graph(e)
        g.ndata["_LINE"] = torch.Tensor(n["id"].astype(int).to_numpy())
        label = self.get_vul_label(idx)
        g.ndata["_LABEL"] = torch.Tensor([label] * len(n))
        g.ndata["_SAMPLE"] = torch.Tensor([_id] * len(n))
        g.ndata["_PAT"] = torch.Tensor([idx >= len(self.df) / 2] * len(n))

        # Add edges between each node and itself to preserve old node representations
        # print(g.number_of_nodes(), g.number_of_edges())
        g = dgl.add_self_loop(g)
        return g


class BigVulDatasetDevignPatch(BigVulDatasetPatch):
    """IVDetect version of BigVul."""

    def __init__(self, **kwargs):
        """Init."""
        super(BigVulDatasetDevignPatch, self).__init__(**kwargs)
        path = get_dir(processed_dir() / kwargs['dataset'] / f"w2v_False")
        self.w2v = MyWord2Vec(path)
        # filter large functions
        print(f'{kwargs["partition"]} LOCAL before large:', len(self.df))
        ret = dfmp(
            self.df,
            BigVulDatasetDevignPatch._feat_ext_itempath,
            "_id",
            ordr=True,
            desc="Cache features: ",
            workers=32
        )
        self.df = self.df[ret]
        print(f'{kwargs["partition"]} LOCAL after large:', len(self.df))

        # Get mapping from index to sample ID.
        print(self.df.columns)

        func_after = self.df['func_after']
        pat = self.df.copy()
        assert len(pat) == len(func_after)
        pat['func_before'] = func_after
        pat['vul'] = 0
        self.df = pd.concat([self.df, pat])
        # print(self.df)
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        print('stats:', self.stats())
        # print(self.df.columns, self.df.shape)
        self.idx2id = pd.Series(self.df._id.values, index=self.df.idx).to_dict()

    def _feat_ext_itempath(_id):
        """Run feature extraction with itempath."""
        code, _, _, _, _, _, counter = devign_feature_extraction(BigVulDatasetPatch.itempath_vul(_id))
        code1, _, _, _, _, _, counter = devign_feature_extraction(BigVulDatasetPatch.itempath_pat(_id))
        return len(code) > 0 and len(code) <= 500 and len(code1) > 0 and len(code1) <= 500

    def get_vul_label(self, idx):
        """Obtain vulnerable or not."""
        return idx < (len(self.df) / 2)

    def __getitem__(self, idx):
        """Override getitem."""
        _id = self.idx2id[idx]
        if idx < len(self.df) / 2:
            code, lineno, nt, ei, eo, et, _ = devign_feature_extraction(BigVulDatasetPatch.itempath_vul(_id))
        else:
            code, lineno, nt, ei, eo, et, _ = devign_feature_extraction(BigVulDatasetPatch.itempath_pat(_id))

        g = dgl.graph((eo, ei))
        # g.ndata["_LINE"] = torch.Tensor(np.array(lineno).astype(int))
        label = self.get_vul_label(idx)
        g.ndata["_LABEL"] = torch.Tensor([label] * len(lineno))
        g.ndata["_SAMPLE"] = torch.Tensor([_id] * len(lineno))

        # node features
        assert g.num_nodes() == len(lineno)
        text_feats = self.w2v.get_embeddings_list(code)
        structure_feats = [type_one_hot[type_map[node_type] - 1] for node_type in nt]
        node_feats = np.concatenate([structure_feats, text_feats], axis=1)

        # debug('node_feats')
        # print('node f', len(node_feats), len(code))
        g.ndata['node_feat'] = torch.Tensor(np.array(node_feats))
        g.ndata["_WORD2VEC"] = torch.Tensor(np.array(node_feats))
        label = self.get_vul_label(idx)
        g.ndata["_LABEL"] = torch.Tensor([label] * len(lineno))
        g.edata["_ETYPE"] = torch.Tensor(np.array(et)).long()
        # Add edges between each node and itself to preserve old node representations
        g = dgl.add_self_loop(g)
        return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bigvul')
    parser.add_argument('--model', type=str, default='ivdetect')
    parser.add_argument('--stat', action='store_true')
    parser.add_argument('--mix', action='store_true')
    parser.add_argument('--not_balance', action='store_true')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    # dataset = 'bigvul' 
    cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    df = pd.read_pickle(cache_path)
    train_df = df[df.partition == 'train']
    valid_df = df[df.partition == 'valid']
    test_df = df[df.partition == 'test']

    if args.model == 'ivdetect':
        testvul_ds = BigVulDatasetIVDetectPatch(df=test_df, partition="test", dataset=dataset)
        # g = testvul_ds.item(174374)
        # # print(g)
        # g = testvul_ds[0]
        # print(g)

        model = IVDetect(input_size=200, hidden_size=128)
    elif args.model == 'devign':
        testvul_ds = BigVulDatasetDevignPatch(df=test_df, partition="test", dataset=dataset)
        model = DevignModel(input_dim=132, output_dim=200)
    elif args.model == 'reveal':
        testvul_ds = BigVulDatasetDevignPatch(df=test_df, partition="test", dataset=dataset)
        # testpat_ds = BigVulDatasetDevignPatch(df=test_df, partition="test", dataset=dataset)
        model = GGNNSum(input_dim=132, output_dim=200)
    elif args.model in ['codebert', 'codet5', 'unixcoder', 'simcl']:
        testvul_ds = BigVulDatasetPatch(df=test_df, partition="test", dataset=dataset)
        # '''
        # DATASET='bigvul'
        # mkdir /data1/username/CLVD/storage/results/codebert/$DATASET
        # CUDA_VISIBLE_DEVICES=1 python3 main.py \
        #   --model_type=roberta \
        #   --tokenizer_name=microsoft/codebert-base \
        #   --model_name_or_path=microsoft/codebert-base \
        #   --do_patch \
        #   --dataset=$DATASET \
        #   --epochs 10 \
        #   --block_size 512 \
        #   --train_batch_size 28 \
        #   --eval_batch_size 16 \
        #   --learning_rate 2e-5 \
        #   --max_grad_norm 1.0 \
        #   --evaluate_during_training \
        #   --seed 123456
        # '''

    id2delta = pd.Series(testvul_ds.df.delta.values, index=testvul_ds.df._id).to_dict()
    # %% Load data
    if args.stat:
        if args.mix:
            cache_path = get_dir(result_dir() / f"patched_result/bigvul_mix/{args.model}")
        else:
            cache_path = get_dir(result_dir() / f"patched_result/bigvul/{args.model}")
        # if args.model in ['codebert', 'codet5', 'unixcoder']:
        #     cache_path = get_dir(cache_path / f"no_balance_{args.not_balance}")
        if args.model == 'reveal':
            a = pickle.load(open(str(cache_path / f'result_True'), 'rb'))
            vul_ids, vul_true, vul_pred = a[0], a[1], a[3]
            print(len(set(vul_ids)), len(vul_true), len(vul_pred), len(vul_true) - sum(vul_true))
            from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1
            print(acc(vul_true, vul_pred),
                  pr(vul_true, vul_pred),
                  rc(vul_true, vul_pred),
                  f1(vul_true, vul_pred), )
            # vul_ids = [i for i in range(int(len(vul_true) / 2))]
            # vul_ids += vul_ids

        else:
            vul_ids, vul_true, vul_logits, vul_pred = pickle.load(open(str(cache_path / f'result_True'), 'rb'))
            print(len(vul_ids), len(vul_true), len(vul_logits), len(vul_pred))
        # pat_true, pat_logits, pat_pred = pickle.load(open(str(cache_path / f'result_False'), 'rb'))
        # debug(f'len vul: {len(vul_pred)}; len pat: {len(pat_pred)}')
        vul_delta = [id2delta[i] for i in vul_ids]
        pair = list(zip(vul_ids, vul_pred, vul_true, vul_delta))
        res = pd.DataFrame(pair, columns=['vul_id', 'vul_pred', 'true_label', 'delta'])
        print(res)
        vul = res[res.true_label == 1]
        vul.columns = ['vul_id', 'vul_pred', 'true_label', 'delta']
        pat = res[res.true_label == 0]
        pat.columns = ['vul_id', 'pat_pred', 'true_label', 'delta']
        print(vul)
        print(pat)
        res = pd.merge(vul, pat, on='vul_id', how='inner')[['vul_id', 'vul_pred', 'pat_pred', 'delta_x']]
        print(res)
        vul_pred_correct = res[res.vul_pred == 1]
        print('recall:', vul_pred_correct.shape[0] / vul.shape[0])
        pat_pred_as_1 = vul_pred_correct.pat_pred.sum()
        pat_pred_as_0 = len(vul_pred_correct) - pat_pred_as_1
        pat_pred_as_1_stat = np.percentile(vul_pred_correct[vul_pred_correct.pat_pred == 1].delta_x.values,
                                           (25, 50, 75), interpolation='midpoint')
        print(pat_pred_as_1_stat, np.mean(vul_pred_correct[vul_pred_correct.pat_pred == 1].delta_x.values))
        pat_pred_as_0_stat = np.percentile(vul_pred_correct[vul_pred_correct.pat_pred == 0].delta_x.values,
                                           (25, 50, 75),
                                           interpolation='midpoint')
        print(pat_pred_as_0_stat, np.mean(vul_pred_correct[vul_pred_correct.pat_pred == 0].delta_x.values))
        debug(
            f'vul_pred_as_1({vul_pred_correct.shape[0]}) = pat_pred_as_1 ({pat_pred_as_1}) + pat_pred_as_0({pat_pred_as_0})')
        debug(f'pat_pred_as_1 / vul_pred_correct = 1:{round(pat_pred_as_1 / vul_pred_correct.shape[0], 4)}')
        show_representation(cache_path, model=args.model, args=args)
        return

    dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
    testvul_dl = GraphDataLoader(testvul_ds, batch_size=64, **dl_args)
    # testpat_dl = GraphDataLoader(testpat_ds, batch_size=64, **dl_args)
    dev = args.device
    model.to(dev)
    # print(testpat_ds, testvul_ds)

    mets = test(model, test_dl=testvul_dl, test_ds=testvul_ds, args=args, vul=True, dev=args.device)
    debug(mets)
    # mets = test(model, test_dl=testpat_dl, test_ds=testpat_ds, args=args, dev=args.device)
    # debug(mets)


def test(model, test_dl, test_ds, args, logger=None, vul=False, dev=None):
    model.eval()
    old_ID = {
        'ivdetect': '202207051706_v1',
        'devign': '202207061711_v1',
        'reveal': '202207061820_v1'
    }
    ID = {
        'ivdetect': '202207081448_v1',
        'devign': '202207081337_v1',
        'reveal': '202207081341_v1',
    }
    mix_ID = {
        'ivdetect': '202207141256_v1',
        'devign': '202207151042_v1',
        'reveal': '202207141352_v1',
    }
    if args.mix:
        path = result_dir() / f"{args.model}/bigvul_mix" / f"{mix_ID[args.model]}/best_f1.model"
        cache_path = get_dir(result_dir() / f"patched_result/bigvul_mix/{args.model}")
    else:
        path = result_dir() / f"{args.model}/bigvul" / f"{ID[args.model]}/best_f1.model"
        cache_path = get_dir(result_dir() / f"patched_result/bigvul/{args.model}")
    model.load_state_dict(torch.load(path))
    with torch.no_grad():
        all_pred = torch.empty((0)).long().to(args.device)
        all_true = torch.empty((0)).long().to(args.device)
        all_ids = torch.empty((0)).long().to(args.device)
        for test_batch in test_dl:
            test_batch = test_batch.to(args.device)
            test_labels = dgl.max_nodes(test_batch, "_LABEL")
            test_ids = dgl.max_nodes(test_batch, "_SAMPLE")
            if args.model == 'reveal':
                _, test_logits = model.save_ggnn_output(test_batch, test_ds)
            else:
                test_logits = model(test_batch, test_ds)
            all_pred = torch.cat([all_pred, test_logits])
            all_true = torch.cat([all_true, test_labels])
            all_ids = torch.cat([all_ids, test_ids])
        if args.model == 'reveal':
            graph_embeddings = all_pred
            model = MetricLearningModel(input_dim=200, hidden_dim=256)
            model.load_state_dict(torch.load(f'../models/reveal/{args.dataset}_best_f1.model'))
            model.to(dev)
            model.eval()
            debug('ggnn')
            print(graph_embeddings.shape)
            dataset = DataSet(128, graph_embeddings.shape[1])
            dataset.clear_test_set()
            for _x, _y, _id in zip(graph_embeddings, all_true, all_ids):
                dataset.add_data_entry(_x.tolist(), _y, _id=_id, part='test')
            # pred = predict(
            #     model=model, iterator_function=dataset.get_next_test_batch,
            #     _batch_count=dataset.initialize_test_batches(), cuda_device=0,
            # )
            print(len(dataset.test_entries))
            print(cache_path)
            tacc, tpr, trc, tf1, pred, all_true, all_ids = evaluate_patch(
                model, dataset.get_next_test_batch, dataset.initialize_test_batches(), 0, None
            )
            pickle.dump((all_ids, all_true, pred, pred),
                        open(str(cache_path / f'result_{vul}'), 'wb'))
            return 'Test Set:       Acc: %6.4f\tF1: %6.4f\tRc %6.4f\tPr: %6.4f' % \
                   (tacc, tf1, trc, tpr)
        if args.model == 'ivdetect':
            all_true = all_true.long()
            test_mets = get_metrics_logits(all_true, all_pred)
            sm_logits = torch.nn.functional.softmax(all_pred.float(), dim=1)
            pos_logits = sm_logits[:, 1].detach().cpu().numpy()
        elif args.model == 'devign':
            test_mets = get_metrics_probs_bce(all_true, all_pred)
            pos_logits = all_pred.detach().cpu().numpy()
        f1_threshold = 0.5
        pred = [1 if i > f1_threshold else 0 for i in pos_logits]
        pickle.dump(
            (all_ids.detach().cpu().numpy(), all_true.detach().cpu().numpy(), all_pred.detach().cpu().numpy(), pred),
            open(str(cache_path / f'result_{vul}'), 'wb'))
    return test_mets


def show_representation(cache_path, model='', args=None):
    ids, representations, labels = pickle.load(open(str(cache_path / f'embeddings_True'), 'rb'))
    print(representations.shape)
    plot_embedding(ids, representations, labels, title=f'{args.dataset}_{args.model}')
    return representations


def plot_embedding(ids, X_org, y, title=None):

    # X, _, Y, _ = train_test_split(X_org, y, test_size=0.5)
    # X, Y = np.asarray(X_org), np.asarray(y)
    X, Y = X_org, y
    # X = X[:10000]
    # Y = Y[:10000]
    # y_v = ['Vulnerable' if yi == 1 else 'Non-Vulnerable' for yi in Y]
    from sklearn import manifold
    import matplotlib.pyplot as plt

    tsne = manifold.TSNE(n_components=2, random_state=0)
    print('Fitting TSNE!')
    X = tsne.fit_transform(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # file_ = open('./tse/' + str(title) + '-tsne-features.json', 'w')
    # if isinstance(X, np.ndarray):
    #     _x = X.tolist()
    #     _y = Y.tolist()
    # else:
    #     _x = X
    #     _y = Y
    # json.dump([_x, _y], file_)
    # file_.close()
    plt.figure(title)
    # sns.scatterplot(X[:, 0], X[:, 1], hue=y_v, palette=['red', 'green'])
    for i in range(X.shape[0]):
        if Y[i] == 0:
            plt.text(X[i, 0], X[i, 1], 'o',
                     fontdict={'weight': 'bold', 'size': 10})
            plt.text(X[i, 0], X[i, 1]*1.05, f'{ids[i]}_{Y[i]}',
                     fontdict={'weight': 'bold', 'size': 6})
        else:
            plt.text(X[i, 0], X[i, 1], '+',
                     color=plt.cm.Set1(0),
                     fontdict={'weight': 'bold', 'size': 10})
            plt.text(X[i, 0], X[i, 1]*1.05, f'{ids[i]}_{Y[i]}',
                     color=plt.cm.Set1(0),
                     fontdict={'weight': 'bold', 'size': 6})
    # plt.scatter()
    # plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title("")
    plt.savefig('./tse/' + title + '.pdf')
    plt.show()


if __name__ == '__main__':
    main()
