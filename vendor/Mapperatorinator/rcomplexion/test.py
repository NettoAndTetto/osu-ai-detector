from pathlib import Path

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from slider import Beatmap

from libs.tokenizer import Tokenizer
from libs.utils import get_model
from libs.tokenizer import Event, EventType


@hydra.main(config_path="configs", config_name="inference_v1", version_base="1.1")
def main(args: DictConfig):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.model_path)
    model_state = torch.load(ckpt_path / "pytorch_model.bin", map_location=device)

    tokenizer = Tokenizer(args)
    model = get_model(args, tokenizer)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)

    max_timeshift = int((args.data.max_time - args.data.min_time) * args.data.time_resolution)
    results = np.empty((max_timeshift + 1, tokenizer.vocab_size_out), dtype=np.float32)

    for i in range(max_timeshift + 1):
        input_ids = torch.tensor([[tokenizer.encode(Event(EventType.TIME_SHIFT, i)), tokenizer.encode(Event(EventType.CIRCLE))] * (args.data.src_seq_len // 2)], device=device)
        output = model(input_ids)
        probs = torch.softmax(output.logits, -1)[0].cpu().numpy()
        results[i] = -probs * np.log2(probs)

    # Plot results as image
    plt.imshow(results, aspect="auto")
    plt.xlabel("Predicted time")
    plt.ylabel("Input time")
    plt.title("Model predictions")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
