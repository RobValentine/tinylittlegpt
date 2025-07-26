import random

def sample_corpus(input_path, output_path, num_lines=10000000, shuffle=True):
    print(f"Reading from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines in corpus: {len(lines):,}")

    if shuffle:
        print(f"Shuffling and sampling {num_lines:,} lines...")
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
        num_lines=config["sample_lines"],  
        shuffle=True
    )
