from tokenizers.implementations import ByteLevelBPETokenizer
import numpy as np
import os
import time
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import tempfile
import shutil

# Allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

# Global tokenizer reference for worker processes
GLOBAL_TOKENIZER = None

def _init_worker(vocab_path, merges_path):
    """Initializer for each worker process."""
    global GLOBAL_TOKENIZER
    GLOBAL_TOKENIZER = ByteLevelBPETokenizer(vocab_path, merges_path)

def _encode_batch_worker(lines):
    """Worker function to encode a batch of lines."""
    encodings = GLOBAL_TOKENIZER.encode_batch(lines)
    return [enc.ids for enc in encodings]

def prepare_dataset(input_path, vocab_path, merges_path, output_path,
                    batch_size=1000, num_workers=None, debug=False):
    start_time = time.time()

    print(f"Loading BPE tokenizer from: {vocab_path} + {merges_path}")

    if debug:
        tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
        total_lines = 0
        total_tokens = 0
        print(f"Debug encoding one line at a time: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'wb') as outfile:
            for raw_line in tqdm(infile, desc="Debug Encoding"):
                total_lines += 1
                line = raw_line.strip()
                if not line:
                    continue
                enc = tokenizer.encode(line)
                ids = np.array(enc.ids, dtype=np.uint16)
                ids.tofile(outfile)
                total_tokens += len(enc.ids)
                print(f"🔹 Line {total_lines}")
                print(f"Original: {line}")
                print(f"Token IDs: {enc.ids}")
                print(f"Tokens:    {enc.tokens}")
        print(f"Processed {total_lines:,} lines")
        print(f"Total tokens: {total_tokens:,}")
        if total_lines > 0:
            print(f"Average tokens per line: {total_tokens / total_lines:.2f}")
        if total_tokens < 1000:
            print("Warning: very small token count. Model may not train effectively.")
        print(f"Saved {total_tokens:,} tokens to {output_path}")
        print(f"Completed in {time.time() - start_time:.2f} seconds.")
        return

    num_workers = num_workers or max(1, cpu_count() - 1)
    print(f"Using {num_workers} worker processes for parallel tokenization")

    file_size = os.path.getsize(input_path)
    total_tokens = 0
    total_lines = 0

    pool = Pool(processes=num_workers,
                initializer=_init_worker,
                initargs=(vocab_path, merges_path))

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_paths = []
        chunk_idx = 0

        try:
            with open(input_path, 'r', encoding='utf-8') as infile, \
                 tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding", bar_format=bar_format) as pbar:

                def batch_generator():
                    nonlocal total_lines
                    batch = []
                    for raw_line in infile:
                        total_lines += 1
                        pbar.update(len(raw_line.encode('utf-8')))
                        line = raw_line.strip()
                        if line:
                            batch.append(line)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    if batch:
                        yield batch

                for enc_ids_list in pool.imap(_encode_batch_worker,
                                              batch_generator(),
                                              chunksize=1):
                    chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.bin")
                    with open(chunk_file, 'wb') as chunk_out:
                        for ids in enc_ids_list:
                            arr = np.array(ids, dtype=np.uint16)
                            arr.tofile(chunk_out)
                            total_tokens += len(ids)
                    chunk_paths.append(chunk_file)
                    chunk_idx += 1

        except KeyboardInterrupt:
            print("\nTraining interrupted. Cleaning up...")
            pool.terminate()
            pool.join()
            raise SystemExit(1)

        pool.close()
        pool.join()

        # Merge all chunks into the final binary
        with open(output_path, 'wb') as outfile:
            for chunk_file in chunk_paths:
                with open(chunk_file, 'rb') as chunk_in:
                    shutil.copyfileobj(chunk_in, outfile)

    # Final stats
    print(f"Processed {total_lines:,} lines")
    print(f"Total tokens: {total_tokens:,}")
    if total_lines > 0:
        print(f"Average tokens per line: {total_tokens / total_lines:.2f}")
    if total_tokens < 1000:
        print("Warning: very small token count. Model may not train effectively.")

    bin_size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved {total_tokens:,} tokens ({bin_size_kb:.1f} KB) to {output_path}")
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    prepare_dataset(
        input_path=config["corpus_path"],
        vocab_path=config["vocab_path"],
        merges_path=config["merges_path"],
        output_path=config["bin_path"],
        batch_size=config.get("batch_size", 1000),
        num_workers=config.get("num_workers", None),
        debug=config.get("debug", False)
    )
