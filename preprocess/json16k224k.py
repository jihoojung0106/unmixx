import json
from pathlib import Path
import glob
import os
json_paths=glob.glob("duet_svs/16k/json/same_song/*.json")
# Load the original JSON file
for json_path in json_paths:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    def replace_16k_with_24k(obj):
        if isinstance(obj, dict):
            return {k: replace_16k_with_24k(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_16k_with_24k(item) for item in obj]
        elif isinstance(obj, str):
            return obj.replace("16k", "24k")
        else:
            return obj
    updated_data = replace_16k_with_24k(data)
    output_path = json_path.replace("/16k/", "/24k/")
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    print(f"Updated JSON saved to: {output_path}")

    output_path
