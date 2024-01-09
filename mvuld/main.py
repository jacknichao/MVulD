# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import sys
path1 = os.path.dirname(sys.path[0])
sys.path.append(path1)
import os

from torchmetrics import F1Score
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import time
import json
import random
import argparse
import datetime
import numpy as np
import torch.nn.functional as F

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

# 计算指标
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
from data import build_loader
from project.MMVD.mmvd.lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor,resume_bestf1_helper,save_bestf1_checkpoint
from data.bigvul_dataset import bigvul_loader_swin


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument('--test', type=int, default=0, help='Train mode=0;Test mode=1')
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--myresume', help='resume from multimodel checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    
    train_data,val_data,test_data,data_loader_train,data_loader_val,data_loader_test,mixup_fn= bigvul_loader_swin(config)
    
    print(f"train length={len(train_data)} val length={len(val_data)} test length={len(test_data)}")
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    # logger.info(str(model)) 

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    # if hasattr(model, 'flops'):
    #     flops = model.flops()
    #     logger.info(f"number of GFLOPs: {flops / 1e9}")
    
    device_ids=[config.LOCAL_RANK]
    print(device_ids)
    # print("----------device_count")
    # print(list(range(torch.cuda.device_count())))

    model.cuda() 

    model_without_ddp = model

    optimizer = build_optimizer(config, model) # optim.AdamW
    
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=list(range(torch.cuda.device_count())), broadcast_buffers=False)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    print(f"--------lr:{config.TRAIN.LR_SCHEDULER.NAME}---------")
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    best_f1 = 0.0
    
    if config.TRAIN.BEST_RESUME: # resume best-f1 modle
        if not os.path.exists(config.OUTPUT):
            os.makedirs(config.OUTPUT)
        resume_file = resume_bestf1_helper(config.OUTPUT) 
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"best-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file 
            config.freeze()
            logger.info(f'best-f1 resuming from {resume_file}') 
        else:
            logger.info(f'no checkpoint found in {config.MODEL.RESUME}, ignoring auto resume')

    if config.MODEL.RESUME: # when pretrain
        print("159:------------------")
        print(config.MODEL.RESUME)
        max_accuracy,epoch = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        # print("=======validation set testing =======")
        # acc1,loss,best_f1 = validate(config, data_loader_val, model)
        # logger.info(f"epoch {epoch} is best_f1 of the network on the {len(val_data)} val images: {best_f1:.3f}%")
        # if config.EVAL_MODE:
        #     return
    
    if config.TRAIN.AUTO_RESUME: 
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
            max_accuracy,epoch = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
            logger.info(f'epoch{epoch} is the last epoch')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')


    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        print("168:------------------")
        print(config.MODEL.PRETRAINED)
        load_pretrained(config, model_without_ddp, logger)
        # acc1, loss = validate(config, data_loader_val, model)
        # logger.info(f"Accuracy of the network on the {len(val_data)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        # throughput(data_loader_test, model, logger)
        throughput(data_loader_val, model, logger)
        return

    if args.test == 0: # do train 
        logger.info("*** Start training ***")
        start_time = time.time()
        ## add early stop
        not_f1_inc_cnt = 0

        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS): #(0,300)
            data_loader_train.sampler.set_epoch(epoch)

            train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler)
            # if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            #     save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
            #                     logger)

            acc1,loss,f1 = validate(config, data_loader_val, model) 
            ## save best-f1 swinV2 modle
            if f1 > best_f1:
                not_f1_inc_cnt = 0
                logger.info("  Best f1: %s", round(f1, 4))
                logger.info("  " + "*" * 20)
                logger.info("[%d] Best f1 changed into %.4f\n" % (epoch, round(f1, 4)))
                best_f1 = f1
                # Save best checkpoint for best f1 ：
                output_dir = os.path.join(config.OUTPUT, 'checkpoint-best-f1')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                max_accuracy = max(max_accuracy, acc1) 
                save_bestf1_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)
            else:
                not_f1_inc_cnt += 1
                logger.info("f1 does not increase for %d epochs", not_f1_inc_cnt)
                if not_f1_inc_cnt > args.patience: 
                    logger.info("Early stop as f1 do not increase for %d times", not_f1_inc_cnt)
                    logger.info("[%d] Early stop as not_f1_inc_cnt=%d\n" % (epoch, not_f1_inc_cnt))
                    is_early_stop = True
                    break
            logger.info(f"Accuracy of the network on the {len(val_data)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            best_f1 = max(best_f1,f1)
            logger.info(f'Max accuracy: {max_accuracy:.3f}%')
            logger.info(f'best f1 : {best_f1:.3f}%')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))
    else: 
        print('\n*** Start testing ***\n')
        acc1, loss,f1 = validate(config, data_loader_test, model)
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
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True) # images
        targets = targets.cuda(non_blocking=True) # (samples,targets)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
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
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    ## add
    start_validate = True
    end = time.time()
    for idx, (images, target) in enumerate(data_loader): 
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True) 
        # print("----------------images shape")
        # print(images.shape) # torch.Size([3, 384, 384])
        # print("validate------------1 target")
        # print(target) # 
        
        ## compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images) 
            prob = F.softmax(output, dim=1) # This is the output probability value of each class
            predict0 = torch.unsqueeze(prob.argmax(dim=1), dim=1)
    
        ## merge output/prob
        if start_validate:
            all_output = output.data.float()
            all_target = target.data.float()
            all_prob = prob.data.float()
            start_validate = False
        else:
            all_output = torch.cat((all_output, output.data.float()), 0)
            all_prob = torch.cat((all_prob, prob.data.float()), 0)
            all_target = torch.cat((all_target, target.data.float()), 0)
            
        ## measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 2)) 
        
        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        # acc5_meter.update(acc5.item(), target.size(0))
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
                # f'Precision {P_meter.val:.3f} ({P_meter.avg:.3f})\t'
                # f'Recall {R_meter.val:.3f} ({R_meter.avg:.3f})\t'
                # f'F1 {F_meter.val:.3f} ({F_meter.avg:.3f})\t'
                # f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    # overall P,R,F
    _, all_predict = torch.max(all_output.data.float(), 1) # predict应该转成了label值
    all_predict_list = all_predict.cpu().numpy()
    all_target_list = all_target.cpu().numpy()
    all_prob_list = all_prob.cpu().numpy()
    
    # print(predict_list)
    cnt,value=count(all_target_list)
    pcnt,pvalue=count(all_predict_list)
    print("---------------------")
    print(f'total length={len(all_target_list)} ,{value} is more ,number={cnt}') 
    print(f'total length={len(all_predict_list)},{pvalue}is more, number={pcnt}') 
    print(all_target_list)
    print(all_predict_list)

    prauc = average_precision_score(all_target_list, all_prob_list[:, 1],pos_label=1) 
    logger.info(f'PRAUC {prauc:.3f}') 
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for number in range(len(all_target_list)):
        if all_predict_list[number] == 1:
            if all_target_list[number] == 1: # 1:vul，0:clean
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

    P = float(TP) / (TP + FP) if (TP + FP != 0) else 0 
    R = float(TP) / (TP + FN) if (TP + FN != 0) else 0 
    F1Score = float((2 * P * R) / (P + R)) if P + R != 0 else 0

    f1 = f1_score(all_target_list, all_predict_list,zero_division=0)
    logger.info(f' * f1 {f1:.3f} ') 
    acc = accuracy_score(all_target_list, all_predict_list)
    # print ("PRECISION is {first} and RECALL is {second} and F-measure is {third}".format(first=P, second=R, third=F))
    # TPR = R
    # TNR = float(TN) / (FP + TN) if (FP + TN != 0) else 0
    
    # logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f' * Acc@1 {acc:.3f} PRECISION {P:.3f} RECALL {R:.3f} F1 {F1Score:.3f} PRAUC {prauc:.3f}')
    return acc1_meter.avg, loss_meter.avg,f1

def count(list):
    mask = np.unique(list)
    tmp = []
    for v in mask:
        tmp.append(np.sum(list==v))
    print(tmp)
    ts = np.max(tmp)
    max_v = mask[np.argmax(tmp)]
    return ts,max_v

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
    print("---------------------")
    # print(os.environ)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK) #
    
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # user NCCL for GPU, use gloo for CPU
    torch.distributed.barrier() 

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

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
    config.TRAIN.BASE_LR = linear_scaled_lr 
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

    main(config)
