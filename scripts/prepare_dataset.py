from tokenizers import Tokenizer
import numpy as np
import os
import time
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

def encode_file_lines(input_path, tokenizer, debug=False):
    token_ids = []

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Encoding {len(lines):,} lines with tokenizer...")
    for i, line in enumerate(tqdm(lines)):
        line = line.strip()
        if line:
            enc = tokenizer.encode(line)
            token_ids.extend(enc.ids)

            if debug:
                decoded_tokens = [tokenizer.id_to_token(tok_id) or f"[{tok_id}]" for tok_id in enc.ids]
                print(f"\n🔹 Line {i + 1}")
                print(f" Original: {line}")
                print(f" Token IDs: {enc.ids}")
                print(f" Tokens:    {decoded_tokens}")

    return token_ids

def prepare_dataset(input_path, tokenizer_path, output_path, debug=False):
    start_time = time.time()

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Reading input corpus: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    num_lines = text.count('\n')
    print(f"Corpus size: {len(text):,} characters across ~{num_lines:,} lines")

    # Encode and get token IDs
    token_ids = encode_file_lines(input_path, tokenizer, debug=debug)
    token_count = len(token_ids)

    print(f"\nTotal tokens: {token_count:,}")
    if num_lines > 0:
        print(f"Average tokens per line: {token_count / num_lines:.2f}")

    if token_count < 1000:
        print("Warning: very small token count. Model may not train effectively.")

    print(f"Saving binary token dataset to: {output_path}")
    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(output_path)

    bin_size = os.path.getsize(output_path) / 1024
    print(f"Done. Saved {token_count:,} tokens ({bin_size:.1f} KB) to {output_path}")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    prepare_dataset(
        input_path=config["corpus_path"],
        tokenizer_path=config["tokenizer_path"],
        output_path=config["bin_path"],
        debug=config["debug"] 
    )
