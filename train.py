import os
import re
import torch
import hashlib
import math
import time
from torch.utils.data import DataLoader
from model.transformer import GPT
from dataset import CodeDataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

def estimated_confidence(loss):
    return 100 * math.exp(-loss)

def sha256sum(filename):
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

# ----- Training config -----
batch_size = config["batch_size"]
block_size = config["block_size"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]
eval_interval = config["eval_interval"]
device =config["device"]

# ----- Load dataset -----
train_path = config["train_path"]
print(f"Loading dataset: {train_path}")
dataset = CodeDataset(train_path, block_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

total_tokens = dataset.total_tokens
steps_per_epoch = total_tokens // (block_size * batch_size)
max_iters = steps_per_epoch * epochs

print(f"""
Training plan:
  Total tokens       : {total_tokens:,}
  Block size         : {block_size}
  Batch size         : {batch_size}
  Epochs             : {epochs}
  Steps per epoch    : {steps_per_epoch:,}
  Total train steps  : {max_iters:,}
""")

# ----- Init model & optimizer -----
print("Initializing model...")
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
step = 0
resume_step = 0
train_hash = sha256sum(train_path)

# ----- Resume from checkpoint if available -----
checkpoints = sorted(
    [f for f in os.listdir(checkpoint_dir) if f.startswith("gpt_model_step_") and f.endswith(".pt")],
    key=lambda f: int(re.search(r"\d+", f).group()), reverse=True
)

resumed = False
for ckpt in checkpoints:
    meta_file = os.path.join(checkpoint_dir, ckpt.replace(".pt", ".meta"))
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            if f.read().strip() == train_hash:
                print(f"Resuming from: {ckpt}")
                full_ckpt = torch.load(os.path.join(checkpoint_dir, ckpt))
                model.load_state_dict(full_ckpt["model"])
                optimizer.load_state_dict(full_ckpt["optimizer"])
                step = int(re.search(r"\d+", ckpt).group())
                resume_step = step
                resumed = True
                break

if not resumed:
    print("Starting fresh.")

start_time = time.time()

# ----- Training loop -----
print("\nBeginning training...")
while step < max_iters:
    for x, y in loader:
        if step >= max_iters:
            break

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            elapsed = time.time() - start_time
            steps_done = step - resume_step
            steps_remaining = max_iters - step
            steps_per_sec = steps_done / max(elapsed, 1e-8)
            eta = steps_remaining / steps_per_sec if steps_per_sec > 0 else float('inf')
            #This is all wildly inaccurate but good enough
            print(f"Step {step:>6} | Loss: {loss.item():.4f} | ~{estimated_confidence(loss):.1f}% confidence | ETA: {eta/60:.1f} min")

        if step % 1000 == 0 and step > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"gpt_model_step_{step}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, checkpoint_path)

            with open(checkpoint_path.replace(".pt", ".meta"), 'w') as f:
                f.write(train_hash)

            print(f"Saved checkpoint: {checkpoint_path}")

        step += 1

# ----- Final save -----
torch.save(model.state_dict(), "gpt_model.pt")
print("\nTraining complete! Final model saved as: gpt_model.pt")
