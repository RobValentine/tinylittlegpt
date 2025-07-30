# TinyLittleGPT

This repository contains a very small GPT style language model written with PyTorch. It includes utilities for preparing datasets, training, and sampling text from the resulting model.
Fair warning, this is not going to be like chatGPT. It's not going to be like Claude. It's not even going to be like grok. This is a very limited GPT I set up to help me learn how the damned things work under the hood (hint: It's all maths).

If you use a large corpus for your data ingestion, it will take a long time to train the tokenizer and an even longer time to train the model itself. This is why people use hyperscale clouds to do training :D
You can get a feel for how it works by just chucking some python code into a txt file (make sure it's reasonably well formatted) and then run through the steps below. It'll output junk but you can at least see how it works.

I've deliberately left full debug output on so you can see the tokenization process, the "surprise" and loss factors and so on. You can turn those off if you just want a clean output.

The v1 versions of things have not had nearly as much love so you may encounter broken things.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package management. Seriously, use uv. It's so much faster. After cloning the repository install `uv`, create a virtual environment and install the dependencies:

```bash
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install torch tokenizers datasets tqdm pandas numpy
```

The dataset preparation scripts also require `pyarrow` for reading parquet files. If you plan to use the ingestion utilities install it as well:

```bash
uv pip install pyarrow
```

## Preparing Data

1. **Ingest code** (optional): The `ingestion/ingest.py` script downloads the dataset configured in `config/config.py`. If `dataset_path` is set it will load from that directory instead of fetching from Hugging Face. The downloaded split is saved to `parquet_path` so the extraction script can process the parquet files.
**Warning:** Large datasets may require significant disk space and download time.

2. **Extract text**: Use `scripts/extract_parquets_to_text.py` to convert downloaded parquet files to a plain text corpus.
3. **Sample subset** (optional): `scripts/sample_corpus_subset.py` allows you to sample a subset of the corpus if the extracted text is very large.

4. **Train tokenizer**: You can train a byte‑level BPE tokenizer using `scripts/train_tokenizer_bpe.py` or a smaller character‑level tokenizer with `scripts/train_tokenizer_character_level.py`.

5. **Create binary dataset**: Convert the text corpus into a binary token dataset using `scripts/prepare_dataset.py` (character level) or `scripts/prepare_dataset_bpe.py` (BPE). The resulting file will be saved as `data/train.bin` or `data/train_bpe.bin`.

## Training

Edit hyperparameters in `config.py` if desired. Then run either of the training scripts:

```bash
python train.py       # for the character‑level dataset
python trainv2.py     # for the BPE dataset
```

Checkpoints are saved in the `checkpoints/` directory and training can resume automatically from the most recent checkpoint.

## Generating Text

After training, generate samples with one of the generation utilities:

```bash
python generate.py --prompt "def hello_world():" --tokens 64 --readable
python generatev2.py --prompt "def hello_world():" --tokens 64 --readable
```

Both scripts load `gpt_model.pt` by default and print top‑k predictions at each step. See the script arguments for options such as temperature, top‑k filtering and writing output to a file.

## Repository Structure

- `model/` – Transformer architecture and layers
- `dataset.py` – Simple dataset loader for tokenized files
- `train.py`, `trainv2.py` – Training scripts (character‑level and BPE)
- `generate.py`, `generatev2.py` – Text generation utilities
- `scripts/` – Helper scripts for tokenizer training and dataset preparation
- `ingestion/` – Example ingestion script for downloading Hugging Face datasets

Enjoy experimenting with this tiny GPT model!

