import sys
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

# Paths
corpus_path = config["corpus_path"]
tokenizer_save_path =  config["tokenizer_path"]
vocab_size = config["vocab_size"]


# Initialize components
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()  # Good for code
tokenizer.decoder = decoders.ByteLevel()

# Trainer
trainer = BpeTrainer(
    vocab_size=vocab_size,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
)

# Train
print("Training BPE tokenizer...")
tokenizer.train(files=[corpus_path], trainer=trainer)

# Optional: Set post-processing template for consistent output
tokenizer.post_processor = TemplateProcessing(
    single="$A",
    pair="$A $B",
    special_tokens=[]
)

# Save
os.makedirs(os.path.dirname(tokenizer_save_path), exist_ok=True)
tokenizer.save(tokenizer_save_path)

print(f"Tokenizer saved to {tokenizer_save_path}")
