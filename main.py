# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time

import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils.storage import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from config import config
from utils.net_utils import constrainScoreByWhole

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def main(config):
    config.defrost()
    config.finetuning = False
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    weight_opt, score_opt = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, opt_l = amp.initialize(model, [weight_opt, score_opt], opt_level=config.AMP_OPT_LEVEL)
    weight_opt = opt_l[0]
    score_opt = opt_l[1]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler1, lr_scheduler2 = build_scheduler(config, weight_opt, score_opt, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.MODEL.RESUME = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, weight_opt, score_opt, lr_scheduler1, lr_scheduler2, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, weight_opt, score_opt, epoch, mixup_fn, lr_scheduler1, lr_scheduler2)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, weight_opt, score_opt, lr_scheduler1, lr_scheduler2, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def master_params(optimizer1, optimizer2):
    """
    Generator expression that iterates over the params owned by ``optimizer``s.

    Args:
        optimizer1: An optimizer previously returned from ``amp.initialize``.
        optimizer2: An optimizer previously returned from ``amp.initialize``.
    """

    for group in optimizer1.param_groups:
        for p in group['params']:
            yield p
    for group in optimizer2.param_groups:
        for p in group['params']:
            yield p

def perform_backward(loss, weight_opt, score_opt):
    if config.AMP_OPT_LEVEL != "O0":
        with amp.scale_loss(loss, [weight_opt, score_opt]) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

def get_grad_norm(model, weight_opt, score_opt):
    if config.AMP_OPT_LEVEL != "O0":
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(master_params(weight_opt, score_opt), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(master_params(weight_opt, score_opt))
    else:
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
    return grad_norm

def calculateGrad(model, fn_avg, fn_list, args):
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            m.scores.grad.data += 1/(args.K-1)*(fn_list[0] - fn_avg)*getattr(m, 'stored_mask_0') + 1/(args.K-1)*(fn_list[1] - fn_avg)*getattr(m, 'stored_mask_1')

def calculateGrad_pge(model, fn_list, args):
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            m.scores.grad.data += 1/args.K*(fn_list[0]*getattr(m, 'stored_mask_0')) + 1/args.K*(fn_list[1]*getattr(m, 'stored_mask_1'))

def train_one_epoch(config, model, criterion, data_loader, weight_opt, score_opt, epoch, mixup_fn, lr_scheduler1, lr_scheduler2):
    model.train()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    v_meter = AverageMeter()
    max_score_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        l, ol, gl, al, a1, a5, ll = 0, 0, 0, 0, 0, 0, 0

        weight_opt.zero_grad()
        score_opt.zero_grad()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        fn_list = []
        for j in range(config.K):
            config.j = j
            outputs = model(samples)
            original_loss = criterion(outputs, targets)
            loss = original_loss/config.K
            fn_list.append(loss.item()*config.K)
            perform_backward(loss, weight_opt, score_opt)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            l = l + loss.item()
            ol = ol + original_loss.item() / config.K
            a1 = a1 + acc1.item() / config.K
            a5 = a5 + acc5.item() / config.K

        fn_avg = l
        if not config.finetuning:
            if config.conv_type == "ReinforceLOO" or config.conv_type == "ReinforceLOOVR" or config.conv_type == "ReinforceLOOVRWeight":
                calculateGrad(model, fn_avg, fn_list, config)
            if config.conv_type == "Reinforce":
                calculateGrad_pge(model, fn_list, config)


        grad_norm = get_grad_norm(model, weight_opt, score_opt)
        weight_opt.step()
        score_opt.step()
        lr_scheduler1.step_update(epoch * num_steps + idx)
        lr_scheduler2.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(l.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        top1.update(a1, targets.size(0))
        top5.update(a5, targets.size(0))
        end = time.time()
        if config.conv_type == "ReinforceLOO" or config.conv_type == "ReinforceLOOVR" or config.conv_type == "Reinforce" or config.conv_type == "ReinforceLOOVRWeight":
            if not config.finetuning:
                with torch.no_grad():
                    constrainScoreByWhole(model, v_meter, max_score_meter)
        if idx % config.PRINT_FREQ == 0:
            lr1 = weight_opt.param_groups[0]['lr']
            lr2 = score_opt.param_groups[0]['lr']
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} w_lr {lr1:.6f} s_lr {lr2:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'top1 {top1.val:.4f} ({top1.avg:.4f})\t'
                f'top5 {top5.val:.4f} ({top5.avg:.4f})\t'
                f'v_meter {v_meter.val:.4f} ({v_meter.avg:.4f})\t'
                f'max_score_meter {max_score_meter.val:.4f} ({max_score_meter.avg:.4f})\t'
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    if config.use_running_stats:
        model.eval()
    with torch.no_grad():
        for idx, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)
            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                )
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

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
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
