import hydra
import lightning
import torch
from omegaconf import DictConfig

from classifier.libs.utils import load_ckpt
from libs import (
    get_dataloaders,
)

torch.set_float32_matmul_precision('high')


@hydra.main(config_path="configs", config_name="train_v1", version_base="1.1")
def main(args: DictConfig):
    model, model_args, tokenizer = load_ckpt(args.checkpoint_path, route_pickle=False)

    _, val_dataloader = get_dataloaders(tokenizer, args)

    if args.compile:
        model.model = torch.compile(model.model)

    trainer = lightning.Trainer(
        accelerator=args.device,
        precision=args.precision,
    )

    trainer.test(model, val_dataloader)


if __name__ == "__main__":
    main()
