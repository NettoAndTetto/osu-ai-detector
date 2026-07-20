import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from omegaconf import DictConfig

from rcomplexion.libs.utils import (
    setup_args,
    train,
    train_profiling,
    get_model,
    get_tokenizer,
    get_scheduler,
    get_optimizer,
    get_dataloaders,
    get_shared_training_state,
)


@hydra.main(config_path="configs", config_name="train_v1", version_base="1.1")
def main(args: DictConfig):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.optim.grad_acc > 1)
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
        gradient_accumulation_steps=args.optim.grad_acc,
        log_with=args.logging.log_with,
        project_config=ProjectConfiguration(
            project_dir="..", logging_dir="tensorboard_logs"
        ),
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.init_trackers(
        "rcomplexion",
        init_kwargs={
            "wandb": {
                "entity": "mappingtools",
                "job_type": "training",
                "config": dict(args),
                "sync_tensorboard": args.profile.do_profile,
                "mode": args.logging.mode,
            }
        }
    )

    setup_args(args)

    shared = get_shared_training_state()
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, args, shared)

    # noinspection PyTypeChecker
    (
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, test_dataloader
    )

    accelerator.register_for_checkpointing(tokenizer)

    if args.checkpoint_path:
        accelerator.load_state(args.checkpoint_path)
        shared.current_train_step = scheduler.scheduler.last_epoch + 1

    if args.compile:
        model = torch.compile(model)

    func = train_profiling if args.profile.do_profile else train

    func(
        model,
        train_dataloader,
        test_dataloader,
        accelerator,
        scheduler,
        optimizer,
        tokenizer,
        args,
        shared,
    )


if __name__ == "__main__":
    main()
