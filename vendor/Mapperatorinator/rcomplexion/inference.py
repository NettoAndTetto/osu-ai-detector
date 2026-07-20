import os
from datetime import timedelta
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from safetensors.torch import load_file
from slider import Beatmap, Slider
from slider.mod import od_to_ms_300
from torch import nn
from tqdm import tqdm

from libs.dataset import OsuParser
from libs.dataset.data_utils import create_sequences, tokenize_events
from libs.tokenizer import Tokenizer
from libs.utils import get_model


def calc_rhythm_complexity(beatmap: Beatmap, model: nn.Module, tokenizer: Tokenizer, parser: OsuParser, device, args):
    leniency = int(od_to_ms_300(beatmap.overall_difficulty) * args.data.time_resolution)
    events = parser.parse(beatmap)
    tokens = tokenize_events(events, tokenizer)
    sequences, labels = create_sequences(tokens, args.data.src_seq_len, tokenizer)

    if len(sequences) == 0:
        return 0

    input_ids = torch.stack(sequences, 0).to(device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    output = model(input_ids)

    logits = output.logits.cpu()
    probs = torch.softmax(logits, dim=-1)
    # entropies = -torch.nn.functional.log_softmax(logits, dim=-1)

    total_loss = 0
    for i, label in enumerate(labels):
        # noinspection PyTypeChecker
        # Add a leniency to the prediction by allowing the model to be off by 3 tokens
        # This accounts for slight misalignments in beatmaps with complex timing
        # The probabilities (before log) in the leniency range should be summed instead of taking the max
        # The leniency should be scaled by the OD of the beatmap
        aggregate_probs = probs[i, label - leniency:label + leniency].sum()
        total_loss += -torch.log(torch.clip(aggregate_probs, 1e-4, 1)).item()
    # total_loss = torch.min(entropies, dim=-1).values.sum().item()

    # Divide the total loss by the drain time to get entropy per second
    break_threshhold = timedelta(milliseconds=5000)
    drain_time = timedelta()
    last_time = None
    for hitobject in beatmap.hit_objects(stacking=False):
        if last_time is not None and hitobject.time - last_time < break_threshhold:
            drain_time += hitobject.time - last_time
        last_time = hitobject.end_time if isinstance(hitobject, Slider) else hitobject.time

    if drain_time.total_seconds() == 0:
        return total_loss

    return total_loss / drain_time.total_seconds()


@hydra.main(config_path="configs", config_name="inference_v1", version_base="1.1")
def main(args: DictConfig):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.model_path)
    if (ckpt_path / "pytorch_model.bin").exists():
        model_state = torch.load(ckpt_path / "pytorch_model.bin", map_location=device)
    else:
        model_state = load_file(ckpt_path / "model.safetensors")

    tokenizer = Tokenizer(args)
    parser = OsuParser(args.data)

    model = get_model(args, tokenizer)
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)

    # Get a list of all beatmap files in the dataset path in the track index range between start and end
    beatmap_files = []
    track_names = ["Track" + str(i).zfill(5) for i in range(0, 16291)]
    # track_names = ["Track" + str(i).zfill(5) for i in range(0, 1000)]
    for track_name in track_names:
        for beatmap_file in os.listdir(
                os.path.join(args.data.train_dataset_path, track_name, "beatmaps"),
        ):
            beatmap_files.append(
                Path(
                    os.path.join(
                        args.data.train_dataset_path,
                        track_name,
                        "beatmaps",
                        beatmap_file,
                    )
                ),
            )

    # Calculate rhythm complexity for each beatmap
    rhythm_complexities = {}
    for beatmap_file in tqdm(beatmap_files, smoothing=0.01):
        beatmap = Beatmap.from_path(beatmap_file)
        rhythm_complexity = calc_rhythm_complexity(beatmap, model, tokenizer, parser, device, args)
        rhythm_complexities[beatmap.beatmap_id] = rhythm_complexity

    # Save rhythm complexities to a spreadsheet
    with open("../datasets/rhythm_complexities.csv", "w") as f:
        for beatmap_id, rhythm_complexity in rhythm_complexities.items():
            f.write(f"{beatmap_id},{rhythm_complexity}\n")


if __name__ == "__main__":
    main()
