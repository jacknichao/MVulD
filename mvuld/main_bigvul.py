from xmlrpc.client import FastMarshaller
import torch as th
import os
import sys
path1 = os.path.dirname(sys.path[0])
sys.path.append(path1)
import dgl
from torchmetrics import F1Score
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import json
import random
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import gc

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from models.new_model import Multi_DefectModel_noFunc,Multi_DefectModel_noGlobalImage

from models.GraphModel import Multi_DefectModel_100,Multi_DefectModel_110,Multi_DefectModel_011,\
    Multi_DefectModel_001,Multi_DefectModel_NOGAT3,Multi_DefectModel_NOGAT,Multi_DefectModel_NOGAT4,Multi_DefectModel_noGraph,\
    Multi_DefectModel_GATPOS,Multi_DefectModel,Multi_DefectModel_NOGAT2,Multi_DefectModel_new_GCN

from models.MotivationModel import Multi_DefectModel_Image,Multi_DefectModel_FuncText,Multi_DefectModel_Graph,\
    Multi_DefectModel_Graph1,Multi_DefectModel_Graph2

import ml
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import get_config
from models import build_model
from data.bigvul_dataset import bigvul_dataset,bigvul_loader_graph
from project.MMVD.mmvd.lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
# from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
#     reduce_tensor
from utils_multi import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor,save_bestf1_checkpoint,resume_bestf1_helper

from IPython.core.ultratb import ColorTB
sys.excepthook = ColorTB()

import warnings
warnings.filterwarnings("ignore")


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--seed', type=int, default=12345,
                        help="random seed for initialization")
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # early stop
    parser.add_argument("--patience", default=50, type=int) 
    parser.add_argument('--test', type=int, default=0, help='Train mode=0;Test mode=1')
    # easy config modification 
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--test_data_path', type=str, help='path to test dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from swin checkpoint') 
    parser.add_argument('--myresume', help='resume from multimodel checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    # Gradient Accumulation
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='my_output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def myMain(config,args):
    # setup_seed() #
    train_data,val_data,test_data,data_loader_train,data_loader_val,data_loader_test,mixup_fn = bigvul_loader_graph(config)
    print(f"train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    ## ----rq1:MMVD---------
    model = Multi_DefectModel_new_GCN(config=config) 
    
    ## ------motivation example modle------
    # model = Multi_DefectModel_Image(config=config)
    # model = Multi_DefectModel_FuncText(config=config)
    # model = Multi_DefectModel_Graph(config=config)
    # model = Multi_DefectModel_Graph1(config=config)
    # model = Multi_DefectModel_Graph2(config=config)

    ## ------rq2-------
    # model = Multi_DefectModel_noGlobalImage(config=config) # no global image
    # model = Multi_DefectModel_noFunc(config=config) # no func text
    # model = Multi_DefectModel_noGraph(config=config) # no graph feature
    
    ## ------rq3-------bbox+gat+gcn
    # model = Multi_DefectModel(config=config) # GAT
    # model = Multi_DefectModel_NOGAT2(config=config) # POS+GCN 101
    # POS+GAT+GCN
    # model = Multi_DefectModel_100(config=config)
    # model = Multi_DefectModel_110(config=config)
    # model = Multi_DefectModel_011(config=config)
    # model = Multi_DefectModel_001(config=config)


    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"number of params: {n_parameters}")
    
    model.cuda()
    model_without_ddp = model
    
    optimizer = build_optimizer(config, model) # optim.AdamW

    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print("frozen params module name:="+ name)
    #     else:
    #         print(" no frozen params module name:="+ name) 
    
    model = torch.nn.parallel.DistributedDataParallel(model, 
    device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    ,find_unused_parameters=True)
    
    cuda = next(model.parameters()).device 
    logger.info(f"train cuda device :{cuda} ")

    loss_scaler = NativeScalerWithGradNormCount() 
    
    print(f"--------lr = {config.TRAIN.LR_SCHEDULER.NAME}---------")
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.: # 0.1
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    best_f1 = 0.0

    if config.TRAIN.BEST_RESUME: # load the best-f1 epoch(modle)
        if not os.path.exists(config.MULTI_OUTPUT):
            os.makedirs(config.MULTI_OUTPUT)
        resume_file = resume_bestf1_helper(config.MULTI_OUTPUT) 
        if resume_file:
            if config.MODEL.MULTI.RESUME:
                logger.warning(f"best-resume changing resume file from {config.MODEL.MULTI.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.MULTI.RESUME = resume_file 
            config.freeze()
            logger.info(f'best-f1 resuming from {resume_file}') 
        else:
            logger.info(f'no checkpoint found in {config.MODEL.MULTI.RESUME}, ignoring auto resume')

    if config.TRAIN.AUTO_RESUME: # load the last epoch
        if not os.path.exists(config.MULTI_OUTPUT):
            os.makedirs(config.MULTI_OUTPUT)
        resume_file = auto_resume_helper(config.MULTI_OUTPUT)
        if resume_file:
            if config.MODEL.MULTI.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.MULTI.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.MULTI.RESUME = resume_file 
            config.freeze()
            logger.info(f'auto resuming from {resume_file}') 
        else:
            logger.info(f'no checkpoint found in {config.MODEL.MULTI.RESUME}, ignoring auto resume')
    
    if config.MODEL.MULTI.RESUME: 
        print(f"======> multi-model Resuming from {config.MODEL.MULTI.RESUME}....................")
        max_accuracy,epoch = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        # acc1, loss,best_f1,prauc = validate(config, data_loader_val, model) 
        # logger.info(f"epoch {epoch} best f1 of the network on the {len(val_data)} val images: {best_f1:.3f}%")
    
    if args.test == 0:
        logger.info("----------- Start training -----------")
        print('\n*** Start training ***\n')
        start_time = time.time()
        ## add early stop
        global_step = 0
        not_f1_inc_cnt = 0
        is_early_stop = False

        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS): 
            # logger.info(f"train epoch :{epoch} ") 
            data_loader_train.sampler.set_epoch(epoch)
            train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler)
            # if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            #     save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
            #                     logger) 

            acc1, loss,f1,prauc = validate(config, data_loader_val, model) 
            ## Save best checkpoint for best f1 ：
            if f1 > best_f1 and prauc!=0: 
                not_f1_inc_cnt = 0
                logger.info("  Best f1: %s", round(f1, 4))
                logger.info("  prauc: %s", round(prauc, 4))
                logger.info("  " + "*" * 20)
                logger.info("[%d] Best f1 changed into %.4f\n" % (epoch, round(f1, 4)))
                best_f1 = f1
                
                output_dir = os.path.join(config.MULTI_OUTPUT, 'checkpoint-best-f1')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if True or args.data_num == -1 :
                    model_to_save = model_without_ddp
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    max_accuracy = max(max_accuracy, acc1) 
                    save_bestf1_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                logger)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best f1 model into %s", output_model_file)
            else:
                not_f1_inc_cnt += 1
                logger.info("f1 does not increase for %d epochs", not_f1_inc_cnt)
                if not_f1_inc_cnt > args.patience: # default=10
                    logger.info("Early stop as f1 do not increase for %d times", not_f1_inc_cnt)
                    logger.info("[%d] Early stop as not_f1_inc_cnt=%d\n" % (epoch, not_f1_inc_cnt))
                    is_early_stop = True
                    break
                # if prauc == 0:
                #     logger.info("prauc is abnormal")
                #     logger.info("[%d] Early stop as not_f1_inc_cnt=%d\n" % (epoch, not_f1_inc_cnt))
                #     break

            logger.info(f"Accuracy of the network on the {len(val_data)} test images: {acc1:.4f}%")
            logger.info(f"loss of the network on the {len(val_data)} test images: {loss:.4f}%")
            # max_accuracy = max(max_accuracy, acc1) 
            logger.info(f'Max accuracy: {max_accuracy:.4f}%')
            best_f1 = max(best_f1,f1)
            logger.info(f'best f1 : {best_f1:.4f}%')

            logger.info("***** CUDA.empty_cache() *****")
            gc.collect()
            torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))
    else: 
        print('\n*** Start testing ***\n')
        acc1, loss,f1,prauc = validate(config, data_loader_test, model)
        logger.info(f"Accuracy of the network on the {len(test_data)} test images: {acc1:.1f}%")


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    
    model.train()
    optimizer.zero_grad() 
    criterion = torch.nn.CrossEntropyLoss() 

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time() 
    for idx, (g,img_embedding,func_text_embedding,target) in enumerate(data_loader):
        ## 1. g
        cuda = next(model.parameters()).device 
        g = g.to(cuda)

        ## imgs：
        # img_batch = [g_i.ndata['_IMG'][0]for g_i in dgl.unbatch(g)]
        # img_batch=torch.stack(img_batch, dim = 0)
        # images = img.cuda(non_blocking=True)
        
        ## 2.img_embedding(swin)
        img_embedding = img_embedding.cuda(non_blocking=True)
        # print(img_embedding.shape) # ([batch_size, 1024])

        ## 3.text embedding：
        func_text_embedding = func_text_embedding.cuda(non_blocking=True)
        targets = target.cuda(non_blocking=True)
        # print(targets) # label 
        
        ## compute output
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(g,img_embedding,func_text_embedding)
            probs = F.softmax(outputs, dim=1) 

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr'] # 
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    
    batch_size=config.DATA.BATCH_SIZE
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    
    start_validate = True
    end = time.time()
    print("===============data loader start")
    for idx, (g,img_embedding,func_text_embedding,target) in enumerate(data_loader): 

        ## 1.g
        cuda = next(model.parameters()).device 
        g = g.to(cuda)
        # print(g.ndata["_ALL_NODE_EMB"].shape) #

        ## 2.img_embedding
        img_embedding = img_embedding.cuda(non_blocking=True)
        # print(img_embedding.shape) 
        
        ## 3.func_text_embedding
        func_text_embedding = func_text_embedding.cuda(non_blocking=True)
        # print(func_text_embedding.shape)

        ## 4.true label
        # print("validate------------1 target")
        targets = target.cuda(non_blocking=True) 
        # print(targets) 
        
        ## compute output
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(g,img_embedding,func_text_embedding)
            probs = F.softmax(outputs, dim=1) 
            # predict0 = torch.unsqueeze(prob.argmax(dim=1), dim=1)
    
        ## merge output/prob
        if start_validate:
            all_output = outputs.data.float()
            all_prob = probs.data.float()
            all_target = targets.data.float()
            start_validate = False
        else:
            all_output = torch.cat((all_output, outputs.data.float()), 0)
            all_prob = torch.cat((all_prob, probs.data.float()), 0)
            all_target = torch.cat((all_target, targets.data.float()), 0)
            
        ## measure accuracy and record loss
        loss = criterion(outputs, targets) # 
        acc1, _ = accuracy(outputs, targets, topk=(1, 2)) # 
        
        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), targets.size(0)) # self.avg = self.sum / self.count
        acc1_meter.update(acc1.item(), targets.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0: # config.PRINT_FREQ=10
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    
    # overall P,R,F 
    # _, all_predict = torch.max(all_output.data.float(), 1) # Dim=1 indicates the maximum value of the output line
    all_predict = all_prob[:, 1] > 0.5 # probability threshould 
    all_predict_list = all_predict.cpu().numpy() 
    all_target_list = all_target.cpu().numpy()
    all_prob_list = all_prob.cpu().numpy()
    
    # print(predict_list)
    cnt,value=count(all_target_list)
    pcnt,pvalue=count(all_predict_list)
    print("---------------------")
    print(all_target_list) 
    print(all_predict_list) 
    
    # vul_lable=1 
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for number in range(len(all_target_list)):
        if all_predict_list[number] == 1:
            if all_target_list[number] == 1: # 1=vul，0=clean
                TP = TP + 1
            else:
                FP = FP + 1 
        elif all_target_list[number] == 1:
            FN = FN + 1
        # else:
        #     if all_target_list[number] == 1:
        #         FN = FN + 1
        #     else:
        #         TN = TN + 1

    P = float(TP) / (TP + FP) if (TP + FP != 0) else 0 # 
    R = float(TP) / (TP + FN) if (TP + FN != 0) else 0 # TPR
    # TPR = R
    # TNR = float(TN) / (FP + TN) if (FP + TN != 0) else 0
    F1Score = float((2 * P * R) / (P + R)) if P + R != 0 else 0

    f1 = f1_score(all_target_list, all_predict_list)
    # logger.info(f' * f1 {F1Score:.3f} ') 
    logger.info(f' * TP {TP:.3f} and (TP+FN) {TP+FN}')

    print(np.isinf(all_prob_list[:, 1]).any()) # fasle
    print(np.isfinite(all_prob_list[:, 1]).all()) # true # all_prob_list[:,1] exists nan，or infinity
    print(np.isnan(all_prob_list[:, 1]).any()) # false

    prauc = 0.0 
    if(np.isfinite(all_prob_list[:, 1]).all()):
        prauc = average_precision_score(all_target_list, all_prob_list[:, 1],pos_label=1) 
    # acc, _ = accuracy(all_output, all_target, topk=(1, 2))
    acc = accuracy_score(all_target_list, all_predict_list) # (y_true, y_pred)

    logger.info(f' * Acc {acc:.3f} PRECISION {P:.3f} RECALL {R:.3f} F1 {F1Score:.3f} PRAUC {prauc:.3f}')
    return acc1_meter.avg, loss_meter.avg, F1Score , prauc 

def count(list):
    mask = np.unique(list)
    tmp = []
    for v in mask:
        tmp.append(np.sum(list==v))
    print(tmp)
    ts = np.max(tmp)
    max_v = mask[np.argmax(tmp)]
    return ts,max_v

if __name__ == '__main__':
    args, config = parse_option()
    args.cuda = args.ngpu > 0 and torch.cuda.is_available() 
    
    # if config.AMP_OPT_LEVEL:
    #     print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
    # print("---------------------")
    # print(os.environ)
    if args.cuda and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    print("----------torch.cuda.is_available-------------")
    torch.cuda.set_device(config.LOCAL_RANK) # local rank for DistributedDataParallel
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier() 
    ## set seed
    seed = config.SEED + dist.get_rank()
    seed = args.seed 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.ngpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # dist.get_world_size() 
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr # linear_scaled_lr=2.5e-06
    print(f"==============> TRAIN.BASE_LR{config.TRAIN.BASE_LR}....................")
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True) 
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump()) 
    logger.info(json.dumps(vars(args)))

    # main(config)
    myMain(config,args)
