import multiprocessing
import time
from multiprocessing.managers import Namespace

import torch
import numpy as np
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)
from transformers import BertForSequenceClassification, BertConfig

from ..dataset import OrsDataset, OsuParser
from ..tokenizer import Tokenizer


def get_shared_training_state() -> Namespace:
    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    shared.current_epoch = 1
    shared.last_log = time.time()
    shared.current_loss = np.Infinity
    shared.best_loss = np.Infinity
    return shared


def get_model(args: DictConfig, tokenizer: Tokenizer) -> torch.nn.Module:
    config_args = {
        "vocab_size": tokenizer.vocab_size_in,
        "num_labels": tokenizer.vocab_size_out,
    }
    if hasattr(args.model, 'overwrite'):
        config_args |= dict(args.model.overwrite)

    if args.model.name == 'bert':
        config = BertConfig(**config_args)
        model = BertForSequenceClassification(config)
    elif args.model.name == 'osu_r':
        from ..model import OsuR
        model = OsuR(args, tokenizer)
    else:
        raise NotImplementedError

    return model


def get_tokenizer(args: DictConfig) -> Tokenizer:
    return Tokenizer(args)


def get_optimizer(model: BertForSequenceClassification, args: DictConfig) -> Optimizer:
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == 'adamw':
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer: Optimizer, args: DictConfig) -> LRScheduler:
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps - args.optim.warmup_steps,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps],
    )

    return scheduler


def get_dataloaders(tokenizer: Tokenizer, args: DictConfig, shared: Namespace) -> tuple[DataLoader, DataLoader]:
    parser = OsuParser(args.data)
    dataset = {
        "train": OrsDataset(
            args.data,
            parser,
            tokenizer,
            shared=shared,
        ),
        "test": OrsDataset(
            args.data,
            parser,
            tokenizer,
            test=True,
            shared=shared,
        ),
    }

    dataloaders = {}
    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=args.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.dataloader.num_workers > 0,
            worker_init_fn=worker_init_fn,
        )

    return dataloaders["train"], dataloaders["test"]


def worker_init_fn(worker_id: int) -> None:
    """
    Give each dataloader a unique slice of the full dataset.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        np.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    )
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
