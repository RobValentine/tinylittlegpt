import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

def sample_corpus(input_path, output_path, num_lines=config["sample_lines"], shuffle=True):
    print(f"Reading from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines in corpus: {len(lines):,}")

    if shuffle:
        print(f"Shuffling and sampling {num_lines} lines...")
        random.shuffle(lines)

    sampled = lines[:num_lines]

    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(sampled)

    print(f"Done. Saved {len(sampled):,} lines.")

if __name__ == "__main__":
    sample_corpus(
        input_path=config["sample_input_path"],
        output_path=config["sample_output_path"],
        num_lines = int(config["sample_lines"].replace("_", "")),
        shuffle=True
    )
