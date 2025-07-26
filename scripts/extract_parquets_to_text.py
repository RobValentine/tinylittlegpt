import os
import glob
import pandas as pd
from tqdm import tqdm

# Configure this dir properly - it'll be similar but the hash will be different
parquet_dir = os.path.expanduser(config["parquet_path"])
output_txt = config["ingestion_output_path"]

# Recursively find all Python parquet files in the cache
def find_python_parquets(base_dir):
    matches = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".parquet") and "python" in root:
                matches.append(os.path.join(root, file))
    return matches

print("Searching for Python parquet files...")
parquet_files = find_python_parquets(parquet_dir)
print(f"Found {len(parquet_files)} parquet files.")

# Extract and write all content fields
with open(output_txt, "w", encoding="utf-8") as out:
    for parquet_file in tqdm(parquet_files, desc="Extracting"):
        try:
            df = pd.read_parquet(parquet_file, engine="pyarrow")
            if "content" in df.columns:
                for line in df["content"].dropna():
                    out.write(line.strip() + "\n")
        except Exception as e:
            print(f"Failed to process {parquet_file}: {e}")

print(f"Extraction complete. Output saved to: {output_txt}")