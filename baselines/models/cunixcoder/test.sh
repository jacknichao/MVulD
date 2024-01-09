DATASET='bigvul_mix'
CUDA=1
CUDA_VISIBLE_DEVICES=$CUDA python3 main.py --do_patch  \
--dataset $DATASET --not_balance
# --not_balance 可以去掉