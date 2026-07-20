import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

from tqdm import tqdm


def main(args):
    path = Path(args.dataset_path)
    diffs = []

    print("Collecting beatmap difficulties...")

    for track in tqdm(path.iterdir()):
        if not track.is_dir():
            continue
        metadata_file = track / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        for beatmap_name in metadata["Beatmaps"]:
            beatmap_metadata = metadata["Beatmaps"][beatmap_name]
            diff = beatmap_metadata["StandardStarRating"]["0"]
            diffs.append(diff)

    # Plot the difficulties as a histogram
    plt.hist(diffs, bins=50)
    plt.xlabel("Difficulty")
    plt.ylabel("Count")
    plt.title("Beatmap Difficulty Distribution")
    plt.show()

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    args = parser.parse_args()

    main(args)
