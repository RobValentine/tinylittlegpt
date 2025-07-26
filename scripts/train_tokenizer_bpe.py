import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config
from tokenizers import ByteLevelBPETokenizer

# Settings
corpus_path = config["corpus_path"]
output_dir =  config["tokenizer_path"]
vocab_size = config["vocab_size"]


os.makedirs(output_dir, exist_ok=True)

# Initialize the tokenizer
tokenizer = ByteLevelBPETokenizer()

print(f"Training tokenizer on: {corpus_path}")
tokenizer.train(
    files=[corpus_path],
    vocab_size=vocab_size,
    min_frequency=1,  # changed
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
    show_progress=True  # not required, but helps!
)

# Save
tokenizer.save_model(output_dir)

print(f"Tokenizer saved to: {output_dir}/")
print("You'll see vocab.json and merges.txt — use these to load the tokenizer later.")
