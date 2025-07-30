#This ingests datsets from huggingface. don't let it run through the whole thing unless you've got lots of time and patience
#5-10 parquets is more than enough. Just hit ctrl+c twice to end it. It caches the files in ~/.cache/huggingface. 
#You will need a huggingface account and will need to generate a token to use this
from datasets import load_dataset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
from config import config

# Determine if a local dataset directory is specified
dataset_kwargs = {}
if config.get("dataset_path"):
    dataset_kwargs["data_dir"] = config["dataset_path"]

# Load the dataset split
ds = load_dataset(config["dataset"], split="train", **dataset_kwargs)

# Save the downloaded dataset so the extraction step can access the parquet files
parquet_dir = os.path.expanduser(config["parquet_path"])
os.makedirs(parquet_dir, exist_ok=True)
ds.to_parquet(parquet_dir)

print(ds)
print(ds[0])

