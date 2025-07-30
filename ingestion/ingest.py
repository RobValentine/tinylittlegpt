#This ingests datsets from huggingface. don't let it run through the whole thing unless you've got lots of time and patience
#5-10 parquets is more than enough. Just hit ctrl+c twice to end it. It caches the files in ~/.cache/huggingface. 
#You will need a huggingface account and will need to generate a token to use this
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

if config['dataset_dir'] is not None:
ds = load_dataset(config["dataset"], data_dir=config["dataset_path"], split="train")

ds = load_dataset(config["dataset"], split="train")

print(ds)
print(ds[0])

