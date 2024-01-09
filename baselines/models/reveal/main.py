import os
import sys
from torch.optim import Adam
from graph_dataset import create_dataset, DataSet
from pathlib import Path
sys.path.append(str((Path(__file__).parent)))
sys.path.append(str((Path(__file__).parent.parent.parent)))
import os
cur_dir = os.getcwd()
pkg_rootdir = os.path.dirname(os.path.dirname(cur_dir))
print(pkg_rootdir)
if pkg_rootdir not in sys.path:
    sys.path.append(pkg_rootdir)

from models.reveal.model import MetricLearningModel
from trainer import train, show_representation
from utils import debug, get_run_id, processed_dir,cache_dir, set_seed, result_dir, get_dir
import numpy as np
import random
import torch
import warnings

warnings.filterwarnings('ignore')
import argparse
from tsne import plot_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir', help='Base Dir',
        default='/data1/username/ReVeal/data/full_experiment_real_data_processed/bigvul/full_graph/balanced/all_balanced'
    )
    parser.add_argument('--dataset', required=True, type=str)
    args = parser.parse_args()
    print(args)
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    base_dir = args.base_dir 
    
    # ggnn_output = torch.load(f'/data1/username/CLVD/storage/cache/ggnn_output/{args.dataset}/ggnn_output.bin')
    ggnn_bin_path_balanced = cache_dir()/f"ggnn_output/{args.dataset}/ggnn_output.bin"
    ggnn_bin_path = cache_dir()/f"ggnn_output/{args.dataset}/not_balance/ggnn_output.bin"
    ggnn_output = torch.load(ggnn_bin_path)

    # train_entries ；valid_entries；test_entries 
    dataset = create_dataset(
        ggnn_output=ggnn_output,
        batch_size=128,
        output_buffer=sys.stderr
    )
    num_epochs = 200
    dataset.initialize_dataset(balance=False) # train entries
    
    # dataset.initialize_dataset(balance=False) # train entries

    ## train_features, train_targets,_ = dataset.prepare_data(
    #     dataset.train_entries, list(range(len(dataset.train_entries)))
    # )
    # plot_embedding(train_features, train_targets, args.name + '-before-training')
    # plot_embedding(train_features, train_targets, args.dataset + '-before-training')
    
    print(dataset.hdim, end='\t')
    model = MetricLearningModel(input_dim=dataset.hdim, hidden_dim=256)
    model.cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    print(model)
    train(model, dataset, optimizer, num_epochs, dataset_name=args.dataset, cuda_device=0, max_patience=10,
          output_buffer=sys.stderr)
    
    ## show_representation(model, dataset.get_next_test_batch, dataset.initialize_test_batches(), 0,
    #                     args.dataset + '-after-training-triplet')
    #
    # model = MetricLearningModel(input_dim=dataset.hdim, hidden_dim=256, lambda1=0, lambda2=0)
    # model.cuda()
    # optimizer = Adam(model.parameters(), lr=0.001)
    # train(model, dataset, optimizer, num_epochs, cuda_device=0, max_patience=10, output_buffer=sys.stderr)
    # show_representation(model, dataset.get_next_test_batch, dataset.initialize_test_batches(), 0,
    #                     args.dataset + '-after-training-no-triplet')
    pass
