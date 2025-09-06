import json
from itertools import combinations
import pandas as pd

# Load JSON data
with open("duet_svs/24k/json/same_song/same_song_dict_OpenSinger_.json", "r") as f:
    data = json.load(f)

# Collect formatted lines
lines = []

for key, entries in data.items():
    # Sort by length (second element of each sublist)
    sorted_entries = sorted(entries, key=lambda x: x[1])
    
    # Generate all 2-combinations
    for e1, e2 in combinations(sorted_entries, 2):
        line = f"{key}|{e1[0]}|{e1[1]}|{e2[0]}|{e2[1]}"
        lines.append(line)

# Save to txt file
output_path = "duet_svs/24k/json/same_song/opensinger_combination.txt"
with open(output_path, "w") as f:
    f.write("\n".join(lines))

output_path
