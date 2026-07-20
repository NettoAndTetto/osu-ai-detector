import glob
import os.path
import time
from multiprocessing.managers import Namespace

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from ..tokenizer import Tokenizer, EventType
from .log_utils import Averager

logger = get_logger(__name__)


def forward(model: BertForSequenceClassification, batch):
    outputs = model(**batch)
    loss = outputs.loss

    stats = {"loss": loss.detach()}
    return loss, stats


def forward_eval(model: BertForSequenceClassification, batch):
    outputs = model(**batch)
    return outputs


def add_prefix(prefix: str, stats: dict[str, float]):
    return {f"{prefix}/{k}": v for k, v in stats.items()}


def maybe_save_checkpoint(accelerator: Accelerator, args: DictConfig, shared: Namespace):
    if (
            shared.current_train_step > args.optim.total_steps
            or shared.current_train_step % args.checkpoint.every_steps == 0
    ):
        if shared.current_loss < shared.best_loss:
            shared.best_loss = shared.current_loss
            is_best = True
        else:
            is_best = False

        output_dir = f"checkpoint-{shared.current_train_step}"
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir=output_dir, safe_serialization=True)

        wandb_tracker = accelerator.get_tracker("wandb")
        if wandb_tracker is not None:
            art = wandb.Artifact(
                f"rcomplexion-{wandb.run.id}",
                type="model",
                metadata={
                    "format": "accelerate",
                    "src_seq_len": args.data.src_seq_len,
                    "current_train_step": shared.current_train_step,
                    "current_epoch": shared.current_epoch,
                    "current_loss": shared.current_loss,
                },
            )

            for file in os.listdir(output_dir):
                art.add_file(os.path.join(output_dir, file))

            wandb.log_artifact(art, aliases=["best"] if is_best else None)
            logger.info(f"Logged checkpoint to wandb: {art.name}")


def maybe_eval(
        model: BertForSequenceClassification,
        accelerator: Accelerator,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        args: DictConfig,
        shared: Namespace,
):
    if (
            shared.current_train_step > args.optim.total_steps
            or shared.current_train_step % args.eval.every_steps == 0
    ):
        model.eval()

        with torch.no_grad():
            eval_model(model, accelerator, dataloader, tokenizer, args, shared)

        shared.last_log = time.time()
        model.train()


def maybe_logging(
        model: BertForSequenceClassification,
        accelerator: Accelerator,
        optimizer: Optimizer,
        averager: Averager,
        args: DictConfig,
        shared: Namespace,
):
    def extra_stats(args, shared, model, optimizer):
        stats = {}

        if args.logging.weights_l2:
            weights_l2 = (
                    sum(p.detach().norm(2).item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
            )
            stats["weights_l2"] = weights_l2

        stats["lr"] = optimizer.param_groups[0]["lr"]
        stats["seconds_per_step"] = (
                                            time.time() - shared.last_log
                                    ) / args.logging.every_steps

        return stats

    if shared.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, shared, model, optimizer)

        averager.update(stats)
        averaged_stats = averager.average()
        averaged_stats["epoch"] = shared.current_epoch
        averaged_stats = add_prefix("train", averaged_stats)
        accelerator.log(averaged_stats, step=shared.current_train_step)
        averaged_stats["step"] = shared.current_train_step
        logger.info(averaged_stats)

        shared.last_log = time.time()


def maybe_grad_clip_and_grad_calc(
        model: BertForSequenceClassification,
        accelerator: Accelerator,
        args: DictConfig,
):
    if args.optim.grad_clip > 0:
        grad_l2 = accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        ).item()
    else:
        grad_l2 = None

    if args.logging.grad_l2:
        if grad_l2 is None:
            grad_l2 = (
                    sum(
                        p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()
                    )
                    ** 0.5
            )

        return {"grad_l2": grad_l2}
    else:
        return {}


# noinspection PyUnresolvedReferences,PyTypeChecker
def eval_model(
        model: BertForSequenceClassification,
        accelerator: Accelerator,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        args: DictConfig,
        shared: Namespace,
):
    shared.last_log = time.time()
    averager = Averager()

    for batch_id, batch in enumerate(dataloader, start=1):
        if batch_id == args.eval.steps * args.optim.grad_acc:
            break

        outputs = forward_eval(model, batch)

        # Reduce loss over all processes
        loss = outputs.loss
        loss = accelerator.reduce(loss, reduction="mean")

        # Gether labels and predictions over all processes and drop duplicates
        preds = torch.argmax(outputs.logits, dim=-1)
        labels = batch["labels"]
        accelerator.gather_for_metrics((preds, labels))

        # Calculate accuracy metrics
        stats = {"loss": loss.detach(),
                 "timing_acc": acc(preds, labels)}

        averager.update(stats)

    averager.update({"time": time.time() - shared.last_log})
    averaged_stats = averager.average()
    averaged_stats = add_prefix("test", averaged_stats)
    accelerator.log(averaged_stats, step=shared.current_train_step)
    logger.info(averaged_stats)

    shared.current_loss = averaged_stats["test/loss"]


def acc(preds, labels):
    accs = preds == labels
    if isinstance(accs, torch.Tensor):
        accs = accs.detach().float().cpu().numpy()
    return accs


def train(
        model: BertForSequenceClassification,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        accelerator: Accelerator,
        lr_scheduler: LRScheduler,
        optimizer: Optimizer,
        tokenizer: Tokenizer,
        args: DictConfig,
        shared: Namespace,
        profiler=None,
):
    model.train()

    train_averager = Averager()

    while shared.current_train_step <= args.optim.total_steps:
        # In case there is a remainder from previous epoch, we need to reset the optimizer
        optimizer.zero_grad(set_to_none=True)

        accelerator.print(f"Epoch {shared.current_epoch}")

        for batch_id, batch in enumerate(train_dataloader, start=1):
            with accelerator.accumulate(model):
                if shared.current_train_step > args.optim.total_steps:
                    break

                loss, stats = forward(model, batch)

                accelerator.backward(loss)
                train_averager.update(stats)

                if accelerator.sync_gradients:
                    stats = maybe_grad_clip_and_grad_calc(model, accelerator, args)
                    train_averager.update(stats)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if profiler is not None:
                    profiler.step()

                if accelerator.sync_gradients:
                    maybe_logging(model, accelerator, optimizer, train_averager, args, shared)
                    maybe_eval(model, accelerator, test_dataloader, tokenizer, args, shared)
                    maybe_save_checkpoint(accelerator, args, shared)

                    shared.current_train_step += 1

        shared.current_epoch += 1

    if not (args.profile.do_profile and args.profile.early_stop):
        maybe_eval(model, accelerator, test_dataloader, tokenizer, args, shared)
        maybe_save_checkpoint(accelerator, args, shared)

    accelerator.end_training()


def train_profiling(
        model: BertForSequenceClassification,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        accelerator: Accelerator,
        lr_scheduler: LRScheduler,
        optimizer: Optimizer,
        tokenizer: Tokenizer,
        args: DictConfig,
        shared: Namespace,
):
    tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(
        "./profiler_logs", worker_name=f"worker_{accelerator.process_index}")

    if args.profile.early_stop:
        stop_step = (args.profile.wait + args.profile.warmup + args.profile.active) * args.profile.repeat / args.optim.grad_acc
        args.optim.total_steps = shared.current_train_step + stop_step

    def on_trace_ready(trace):
        tensorboard_trace_handler(trace)
        wandb_tracker = accelerator.get_tracker("wandb")
        if wandb_tracker is not None:
            wandb.save(glob.glob(f"./profiler_logs/*.pt.trace.json")[0], base_path="profiler_logs")

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=args.profile.wait,
                warmup=args.profile.warmup,
                active=args.profile.active,
                repeat=args.profile.repeat,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
    ) as p:
        train(
            model,
            train_dataloader,
            test_dataloader,
            accelerator,
            lr_scheduler,
            optimizer,
            tokenizer,
            args,
            shared,
            p
        )
