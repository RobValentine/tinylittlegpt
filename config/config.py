config = {
    "debug": False, # Set False when you're not inspecting stuff. Will output a LOT

    #ingestion parameters
    "parquet_path":"~/.cache/huggingface/hub/datasets--bigcode--the-stack/snapshots/349a71353fd5868fb90b593ef09e311379da498a",
    "ingestion_output_path":"data/stack_python_extracted.txt",  # Path to save all extracted Python from The Stack

    "sample_input_path":"data/stack_python_extracted.txt",  # The big extracted file to sample from
    "sample_output_path":"data/python_subset_10m.txt",  # Output of the sampled subset
    "sample_lines":"10_000_000", # You can change this to 20_000_000, etc. for larger sample sizes from your overall corpus

    "vocab_size": 50257,       # Will match tokenizer
    "context_length": 512,     # How many tokens it can look at
    "embedding_dim": 768,     # Size of the vector used to represent each token. Bigger = more expressive, but slower.
    "num_heads": 8,           # Number of attention heads in each transformer block. More heads = better multitasking.
    "num_layers": 6,          # Number of stacked transformer blocks (i.e., model depth). More layers = more learning capacity.
    "ffn_hidden_dim": 2048,    # Size of MLP hidden layer
    "dropout": 0.1,           # Randomly drops 10% of neurons during training to prevent overfitting (i.e., helps generalize).
    "block_size": 512,         # Size of input blocks for training
    "corpus_path":"data/tiny_python_code.txt",# Change this to whatever corpus yhou're using. This is just for my small test sample to check shit isn't broken
    "tokenizer_path":"tokenizer_test",  # Path to tokenizer.json or similar
    "bin_path":"data/train_test.bin",  # Path to output binary token file for training
    "vocab_path":"tokenizer_test/vocab.json",  # Path to vocab file if you're loading from merges/vocab
    "merges_path":"tokenizer_test/merges.txt", # Path to merges file

    # Training-specific parameters
    "batch_size": 16,  # Number of training examples per batch (higher = faster but uses more GPU RAM)
    "learning_rate": 3e-4,  # Base learning rate for optimizer
    "epochs": 3,  # Full passes over dataset
    "eval_interval": 250,  # How often to print loss/checkpoint during training

    # Environment and I/O
    "device": "cuda",  # Use GPU if CUDA is available. If not, use 'cpu'
    "train_path": "data/train_test.bin",  # Input file of encoded tokens for training
    "final_model_path": "gpt_model_test.pt",  # Output file to save the final trained model    
    "checkpoint_dir": "checkpoints",  # Directory to save model checkpoints every N steps

    #Generation Settings
    "gen_model_path": "gpt_model_test.pt"
}