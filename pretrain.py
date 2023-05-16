import wandb

import os
from typing import Iterable
from pathlib import Path
import time
import datetime
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import RandomSampler

from dataset import create_tokenizer, create_dataset, create_dataloader, vit_transform_randaug

from loguru import logger

from model import MPlugConfig, MPlug, MPlugProcessor, MPlugForImageCaption

from utils import misc

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0 
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens proces

def train_one_epoch(
    device: torch.device,
    dataloader: Iterable,
    val_dataloader: Iterable,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accum_iter=1,
    train_state=TrainState(),
    run=None,
    do_log=False,
):
    """Train a single epoch"""
    model.train(True)
    optimizer.zero_grad()

    batch_losses = []
    n_accum = 0

    for iter, batch in enumerate(dataloader):
        logger.info(f"Batch {iter}")
        image, prefix_text, decoder_input_text, label_text, _ = batch.values()
        image, prefix_text, decoder_input_text, label_text = (
            image.to(device), prefix_text.to(device), decoder_input_text.to(device), label_text.to(device)
        ) 
        train_loss = model(image, prefix_text, decoder_input_text, label_text)
        
        train_loss.backward()

        train_state.step += 1
        train_state.samples += image.shape[0]
        
        if iter % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        scheduler.step()

        batch_losses.append(float(train_loss.cpu()))

        if do_log:
            run.log({"train_loss": batch_losses[-1], 
                     "lr": float(optimizer.param_groups[0]["lr"])})
        

    for i, val_batch in enumerate(val_dataloader):
        image, prefix_text, decoder_input_text, label_text, _ = val_batch.values()
        image, prefix_text, decoder_input_text, label_text = (
            image.to(device), prefix_text.to(device), decoder_input_text.to(device), label_text.to(device)
        ) 
        
        with torch.no_grad():
            val_loss = model(image, prefix_text, decoder_input_text, label_text)

        if do_log:
            run.log({'val_loss': float(val_loss.cpu())})
        
    loss = sum(batch_losses) / len(batch_losses)

    return loss, train_state

def main(args, run=None):
    torch.cuda.set_device(1)

    do_log = run is not None

    config = MPlugConfig()
    config.freeze_image = False
    config.freeze_text = False

    logger.debug('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    logger.debug("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    dataset_train = create_dataset(
        dataset_names=args.dataset_names,
        split='train',
        processor=MPlugProcessor(
                create_tokenizer('PhobertTokenizer'),
            ),
        transforms=vit_transform_randaug,
        config=config,
        seed=seed
    )
    
    dataset_val = create_dataset(
        dataset_names=args.dataset_names,
        split='val',
        processor=MPlugProcessor(
                create_tokenizer('PhobertTokenizer'),
            ),
        transforms=vit_transform_randaug,
        config=config,
        seed=seed
    )

    sampler_train = RandomSampler(dataset_train)

    dataloader_val = create_dataloader(
        batch_size=args.batch_size,
        dataset=dataset_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    dataloader_train = create_dataloader(
        batch_size=args.batch_size,
        dataset=dataset_train,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    model = MPlugForImageCaption(config=config)

    model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None: # only base_lr is specified
        args.lr = args.base_lr * eff_batch_size / 256

    logger.info("base_lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logger.info("actual_lr: %.2e" % args.lr)

    logger.info("accumulate grad iterations: %d" % args.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    # optimizer
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999))

    # lr_scheduler ued in transformer
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: misc.rate(
            step, 10, factor=1, warmup_steps=args.warmup_steps
        ),
    )

    # criterion copied from http://nlp.seas.harvard.edu/annotated-transformer/#loss-computation
    criterion = misc.LabelSmoothing(config.vocab_size, config.pad_token_id, smoothing=0.1)
    loss_compute = misc.SimpleLossCompute(criterion=criterion)

    # if run.resumed:
    #     checkpoint = torch.load(
    #         args.checkpoint_path,
    #         map_location='cpu'
    #     )
    #     model.load_state_dict(checkpoint['model'])
    #     # args.start_epoch = checkpoint['epoch']
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    
    logger.info(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    save_dir = os.path.join(args.output_dir, args.run_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch {epoch}")
        batch_losses, _ = train_one_epoch(
            device, 
            dataloader_train, 
            dataloader_val, 
            model, 
            optimizer, 
            lr_scheduler, 
            accum_iter=args.accum_iter, 
            run=run, 
            do_log=do_log
        )

        if epoch % args.print_every == 0:
            logger.info(f"Epoch{epoch}: loss = {batch_losses}")

        if epoch % args.save_every == 0:
            misc.save_model(
                args, 
                epoch, 
                batch_losses, 
                model, 
                optimizer, 
                lr_scheduler=lr_scheduler, 
                save_dir=save_dir
            )     

    del model

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimVLM pre-training', add_help=False)
    # File root
    parser.add_argument('--data_root', default='', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./result', type=str,
                        help='Path where to save outputs')
    parser.add_argument('--checkpoint_path', default='')

    # Optimizer params
    parser.add_argument('--lr', default=None, type=float, metavar='LR',
                        help='Learning rate')
    parser.add_argument('--base_lr', default=5e-4, type=float, metavar='LR',
                        help='Base learning rate')
    parser.add_argument('--warmup_steps', default=3000, type=int, metavar='N',
                        help='Steps to warm up lr')
    
    # Data
    parser.add_argument('--dataset_names', default=['custom'])
    parser.add_argument('--batch_size', default=2, type=int,  # TODO: Fix this later
                        help="Batch size per gpu")
    parser.add_argument('--random_prefix_len', default=False, type=bool,
                        help='is prefix text length randomly chosen')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    
    # Training params
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to pretrain')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--print_every', default=1, type=int,
                        help='number of epochs of interval to save checkpoints')
    parser.add_argument('--save_every', default=1, type=int,
                        help='number of epochs of interval to save checkpoints')
    
    # wandb
    parser.add_argument('--run_name', default='', type=str)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--run_id', default=None)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # if args:
    #     run = wandb.init(
    #         project="simvlm",
    #         name=args.run_name,
    #         dir=args.output_dir,
    #         config={
    #             'datasets': args.dataset_names,
    #             "base_lr": args.base_lr,
    #             "warmup_steps": args.warmup_steps,
    #             "batch_size": args.batch_size,
    #             "epochs": args.epochs,
    #             "accum_iter": args.accum_iter,
    #             "seed": args.seed,
    #             # 'random_prefix_len': args.random_prefix_len
    #         },
    #         # resume=args.resume,
    #         # id=args.run_id
    #     )
    # else:
    run = None
    try:
        main(args, run)
    except:
        # torch.cuda.empty_cache()
        logger.exception('Something wrong')
        exit(code=-1)
    # wandb.finish()