config = {
    "debug": False, # Set False when you're not inspecting stuff. Will output a LOT

    #ingestion parameters    
    "dataset":"storytracer/LoC-PD-Books",  # Hugging Face dataset name
    "dataset_path": None,  # Set to a local directory to load the dataset from disk
    "parquet_path":"~/.cache/huggingface/hub/datasets--storytracer--LoC-PD-Books/snapshots/5f1f4ee054362bcddbb56640c6beb67a92a0bf41/data",  # Directory where downloaded parquet files will be stored
    "ingestion_output_path":"data/stack_books_extracted.txt",  # Path to save all extracted Python from The Stack
    "language_filter": None,  # Set to None (the value, not a string) to extract from all langs
    "content_column": "text",  # Change if your parquet format differs
    "num_threads": 16,  # Number of threads to use for extraction

    "sample_input_path":"data/stack_books_extracted.txt",  # The big extracted file to sample from
    "sample_output_path":"data/books_subset_10m.txt",  # Output of the sampled subset
    "sample_lines":"10_000_000", # You can change this to 20_000_000, etc. for larger sample sizes from your overall corpus

    #Tokenizer parameters
    "vocab_size": 50257,       # Will match tokenizer
    "context_length": 512,     # How many tokens it can look at
    "embedding_dim": 768,     # Size of the vector used to represent each token. Bigger = more expressive, but slower.
    "num_heads": 8,           # Number of attention heads in each transformer block. More heads = better multitasking.
    "num_layers": 6,          # Number of stacked transformer blocks (i.e., model depth). More layers = more learning capacity.
    "ffn_hidden_dim": 2048,    # Size of MLP hidden layer
    "dropout": 0.1,           # Randomly drops 10% of neurons during training to prevent overfitting (i.e., helps generalize).
    "block_size": 512,         # Size of input blocks for training
    #"corpus_path":"data/stack_books_extracted.txt",# Change this to whatever corpus yhou're using. This is just for my small test sample to check shit isn't broken
    "corpus_path":"data/books_subset_10m.txt",
    "tokenizer_path":"tokenizer_bpe",  # Path to tokenizer.json or similar
    "bin_path":"data/train_books_bpe.bin",  # Path to output binary token file for training
    "vocab_path":"tokenizer_bpe/vocab.json",  # Path to vocab file if you're loading from merges/vocab
    "merges_path":"tokenizer_bpe/merges.txt", # Path to merges file


    # Training-specific parameters
    "batch_size": 16,  # Number of training examples per batch (higher = faster but uses more GPU RAM)
    "learning_rate": 2e-5,  # Base learning rate for optimizer
    "epochs": 3,  # Full passes over dataset
    "eval_interval": 250,  # How often to print loss/checkpoint during training
    "num_workers": 6,  # Number of worker processes for data loading

    # Environment and I/O
    "device": "cuda",  # Use GPU if CUDA is available. If not, use 'cpu'
    "train_path": "data/train_books_bpe.bin",  # Input file of encoded tokens for training
    "final_model_path": "gpt_book_model.pt",  # Output file to save the final trained model    
    "checkpoint_dir": "checkpoints",  # Directory to save model checkpoints every N steps

    #Generation Settings
    "gen_model_path": "gpt_book_model.pt",    
    "final_model_path": "gpt_book_model.pt",

    # Reproducibility
    "seed": 42,            # or e.g. 42
    "deterministic": True,  # torch.backends.cudnn.deterministic

    # Checkpointing
    "checkpoint_interval": 1000,  # steps between saves
    "max_checkpoints": 5          # how many old ckpts to keep
}
