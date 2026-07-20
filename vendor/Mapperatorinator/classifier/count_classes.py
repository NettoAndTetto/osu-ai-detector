import json
from pathlib import Path


def init_mapper_idx(mappers_path):
    """"Indexes beatmap mappers and mapper idx."""
    path = Path(mappers_path)

    if not path.exists():
        raise ValueError(f"mappers_path {path} not found")

    # Load JSON data from file
    with open(path, 'r') as file:
        data = json.load(file)

    # Populate beatmap_mapper
    beatmap_mapper = {}
    for item in data:
        beatmap_mapper[item['id']] = item['user_id']

    # Get unique user_ids from beatmap_mapper values
    unique_user_ids = list(set(beatmap_mapper.values()))

    # Create mapper_idx
    mapper_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    num_mapper_classes = len(unique_user_ids)

    return beatmap_mapper, mapper_idx, num_mapper_classes


path = "../datasets/beatmap_users.json"
beatmap_mapper, mapper_idx, num_mapper_classes = init_mapper_idx(path)

print("Number of mapper classes:", num_mapper_classes)
print("Number of beatmaps:", len(beatmap_mapper))
# Calculate number of maps per mapper
maps_per_mapper = {}
for beatmap_id in beatmap_mapper:
    user_id = beatmap_mapper[beatmap_id]
    if user_id not in maps_per_mapper:
        maps_per_mapper[user_id] = 0
    maps_per_mapper[user_id] += 1

# Calculate average maps per mapper class
average_maps_per_mapper = len(beatmap_mapper) / num_mapper_classes
print("Average maps per mapper class:", average_maps_per_mapper)

# Calculate median maps per mapper class
median_maps_per_mapper = sorted(maps_per_mapper.values())[num_mapper_classes // 2]
print("Median maps per mapper class:", median_maps_per_mapper)

# Mapper with most number of maps
max_maps = max(maps_per_mapper.values())
max_maps_mapper = [user_id for user_id in maps_per_mapper if maps_per_mapper[user_id] == max_maps]
print("Mapper with most number of maps:", max_maps_mapper)
print("Number of maps:", max_maps)
