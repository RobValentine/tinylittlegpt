import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

# Config setup
parquet_dir = os.path.expanduser(config["parquet_path"])
output_txt = config["ingestion_output_path"]
content_column = config.get("content_column", "content")
language_filter = config.get("language_filter", None)
max_workers = config.get("num_threads", 8)

# Thread-safe write
write_lock = threading.Lock()
extracted_lines = 0
missing_column_info = {}

# Locate .parquet files
def find_parquets(base_dir):
    matches = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".parquet"):
                if not language_filter or language_filter.lower() in root.lower():
                    matches.append(os.path.join(root, file))
    return matches

print(f"Searching for parquet files (filter: {language_filter})...")
parquet_files = find_parquets(parquet_dir)
print(f"Found {len(parquet_files)} parquet files.")

# Worker function
def process_parquet(parquet_file):
    local_lines = []
    try:
        df = pd.read_parquet(parquet_file, engine="pyarrow")
        if content_column in df.columns:
            for line in df[content_column].dropna().astype(str):
                cleaned = line.strip()
                if cleaned:
                    local_lines.append(cleaned + "\n")
            return (parquet_file, local_lines, None)
        else:
            return (parquet_file, [], list(df.columns))  # missing column
    except Exception as e:
        return (parquet_file, [], f"Error: {e}")

# Parallel execution
with open(output_txt, "w", encoding="utf-8") as out_file, ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_parquet, f) for f in parquet_files]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
        parquet_file, lines, issue = future.result()
        if issue is None:
            with write_lock:
                out_file.writelines(lines)
                extracted_lines += len(lines)
        elif isinstance(issue, list):
            missing_column_info[parquet_file] = issue
        else:
            print(f"Failed to process {parquet_file}: {issue}")

print(f"\nExtraction complete. {extracted_lines:,} lines saved to: {output_txt}")

if missing_column_info:
    print("\nSome files did not contain the expected column:")
    for file, cols in missing_column_info.items():
        print(f"  - {file}: Available columns = {cols}")
