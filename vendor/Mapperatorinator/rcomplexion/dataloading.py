import multiprocessing

import hydra
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from libs.dataset import OsuParser, OrsDataset
from libs.utils import (
    setup_args,
    get_tokenizer,
    worker_init_fn,
)


@hydra.main(config_path="configs", config_name="train_v1", version_base="1.1")
def main(args: DictConfig):
    setup_args(args)

    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    tokenizer = get_tokenizer(args)
    parser = OsuParser(args.data)
    dataset = OrsDataset(
        args.data,
        parser,
        tokenizer,
        shared=shared,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.optim.batch_size,
        num_workers=args.dataloader.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.dataloader.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    if args.mode == 'benchmark':
        # Make histogram of the lengths of the sequences
        lengths = []
        for b in tqdm.tqdm(dataloader, smoothing=0.01):
            ...
            # for i in range(len(b["decoder_input_ids"])):  # batch size
            #     length = b['decoder_input_ids'][i].shape[0]
            #     lengths.append(length)
            # shared.current_train_step += 1
            # if len(lengths) > 10000:
            #     break

        plt.hist(lengths, bins=100)
        plt.show()

        print(f"Max length: {max(lengths)}")
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {sum(lengths) / len(lengths)}")
        print(f"Median length: {sorted(lengths)[len(lengths) // 2]}")
        print(f"75th percentile: {sorted(lengths)[len(lengths) * 3 // 4]}")
        print(f"90th percentile: {sorted(lengths)[len(lengths) * 9 // 10]}")
        print(f"95th percentile: {sorted(lengths)[len(lengths) * 19 // 20]}")
        print(f"99th percentile: {sorted(lengths)[len(lengths) * 99 // 100]}")
        print(f"99.9th percentile: {sorted(lengths)[len(lengths) * 999 // 1000]}")
        print(f"99.99th percentile: {sorted(lengths)[len(lengths) * 9999 // 10000]}")
        print(f"99.999th percentile: {sorted(lengths)[len(lengths) * 99999 // 100000]}")

        print(f"Total number of sequences: {len(lengths)}")
        print(f"Total number of tokens: {sum(lengths)}")
        print(f"Total number of sequences with length 0: {lengths.count(2)}")

    if args.mode == 'plot':
        for b in tqdm.tqdm(dataloader, smoothing=0.01):
            input_times = b['input_ids'].detach().cpu().numpy()[:, ::2]
            label_times = b['labels'].detach().cpu().numpy()
            img = np.concatenate([input_times, label_times], axis=1)
            plt.imshow(img)
            plt.show()
            break

if __name__ == "__main__":
    main()
