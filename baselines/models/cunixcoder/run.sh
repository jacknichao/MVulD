DATASET='bigvul'
CUDA=0,3

mkdir -p /data1/username/project/MMVD/baselines/storage/results/unixcoder/$DATASET
# mkdir -p /data/xinrongguo/linevd/multi_model_baselines/storage/cache/data/$DATASET

CUDA_VISIBLE_DEVICES=$CUDA python3 main.py --do_train --do_test  --do_eval \
--dataset $DATASET 2>&1 | tee /data1/username/project/MMVD/baselines/storage/results/unixcoder/$DATASET/train.log
