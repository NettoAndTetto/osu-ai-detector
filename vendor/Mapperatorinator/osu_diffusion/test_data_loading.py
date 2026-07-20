import hydra
import torch
from tqdm import tqdm

from utils.positional_embedding import position_sequence_embedding
from utils.data_loading import get_data_loader
from utils.tokenizer import Tokenizer


@hydra.main(config_path="../configs/diffusion", config_name="v1", version_base="1.1")
def main(args):
    tokenizer = Tokenizer(args)
    dataloader = get_data_loader(
        args=args,
        tokenizer=tokenizer,
        pin_memory=False,
        drop_last=True,
    )

    if args.mode == "plotfirst":
        import matplotlib.pyplot as plt

        for (x, c), y in dataloader:
            x = torch.swapaxes(x, 1, 2)  # (N, T, C)
            c = torch.swapaxes(c, 1, 2)  # (N, T, E)
            print(x.shape, c.shape, y.shape)
            batch_pos_emb = position_sequence_embedding(x * 512, 128)
            print(batch_pos_emb.shape)
            print(y)

            for j in range(args.optim.batch_size):
                fig, axs = plt.subplots(2, figsize=(5, 5))
                axs[0].imshow(batch_pos_emb[j])
                axs[1].imshow(c[j])
                print(y[j])
                plt.show()
            break
    elif args.mode == "benchmark":
        for _ in tqdm(dataloader, total=7000, smoothing=0.01):
            pass


if __name__ == "__main__":
    main()
