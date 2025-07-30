import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

def sample_corpus(input_path, output_path, num_lines=config["sample_lines"], shuffle=True):
    """Sample ``num_lines`` from ``input_path`` without loading the full file."""

    num_lines = int(str(num_lines).replace("_", ""))
    print(f"Reading from: {input_path}")

    reservoir = []
    total_lines = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if len(reservoir) < num_lines:
                reservoir.append(line)
            else:
                j = random.randint(0, total_lines - 1)
                if j < num_lines:
                    reservoir[j] = line

    print(f"Total lines in corpus: {total_lines:,}")

    if shuffle:
        print(f"Shuffling {len(reservoir):,} sampled lines...")
        random.shuffle(reservoir)

    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(reservoir)

    print(f"Done. Saved {len(reservoir):,} lines.")

if __name__ == "__main__":
    sample_corpus(
        input_path=config["sample_input_path"],
        output_path=config["sample_output_path"],
        num_lines = int(config["sample_lines"].replace("_", "")),
        shuffle=True
    )
