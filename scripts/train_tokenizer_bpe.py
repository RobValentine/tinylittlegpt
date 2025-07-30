import sys
import os
import signal
from tokenizers import ByteLevelBPETokenizer

# 1) Restore default Ctrl+C handler
signal.signal(signal.SIGINT, signal.default_int_handler)

# 2) Load your config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "..", "config")))
from config import config

corpus_path = config["corpus_path"]
output_dir  = config["tokenizer_path"]
vocab_size  = config["vocab_size"]

os.makedirs(output_dir, exist_ok=True)

# 3) Initialize & train (streaming from disk!)
tokenizer = ByteLevelBPETokenizer()

print(f"Training tokenizer on: {corpus_path}")
try:
    tokenizer.train(
        files=[corpus_path],         # <-- point it at your file
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
        show_progress=True           # <-- Rust’s own progress bar
    )
except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    sys.exit(1)

# 4) Save
tokenizer.save_model(output_dir)
print(f"Tokenizer saved to: {output_dir}/")
print("You'll see vocab.json and merges.txt — use these to load the tokenizer later.")
