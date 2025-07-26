import os
import re
import torch
import hashlib
import math
import time
import json
from torch.utils.data import DataLoader
from model.transformer import GPT
from dataset import CodeDataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

# Percentage doesn't really make sense here but I prefer it to remind me how low the confidence actually is.
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
device = config["device"]
train_path = config["train_path"]
final_model_path = config["final_model_path"]
checkpoint_dir = config["checkpoint_dir"]

# ----- Pre-check: Predict training time if possible -----
os.makedirs("benchmarks", exist_ok=True)
benchmark_file = os.path.join("benchmarks", f"benchmark_{os.path.basename(train_path)}.json")

predicted_time = None
if os.path.exists(benchmark_file):
    with open(benchmark_file, 'r') as f:
        previous = json.load(f)
    if config.get("debug"):
        print(f"[DEBUG] Loaded benchmark file: {benchmark_file}")
        print(f"[DEBUG] Previous benchmark contents: {previous}")
    steps_recorded = previous.get("steps_recorded", 0)
    duration = previous.get("duration_sec", 0)
    if (
        duration > 0 and steps_recorded > 0 and
        previous.get("block_size") == block_size and
        previous.get("batch_size") == batch_size and
        previous.get("epochs") == epochs
    ):
        steps = previous.get("steps", [])
        avg_time_per_step = duration / max(1, len(steps))
        total_tokens_est = os.path.getsize(train_path) * 2
        steps_per_epoch_est = max(1, total_tokens_est // (block_size * batch_size))
        max_iters_est = max(1, steps_per_epoch_est * epochs)
        predicted_time = avg_time_per_step * max_iters_est

if predicted_time:
    print(f"Estimated training time: ~{predicted_time / 60:.1f} minutes")
else:
    print("Estimated training time: Unknown (no prior benchmark)")

user_input = input("Do you want to continue training? [Y/n] ").strip().lower()
if user_input == 'n':
    print("Aborting training.")
    exit()

# ----- Check if model already exists -----
if os.path.exists(final_model_path):
    overwrite = input(f"Model file {final_model_path} already exists. Overwrite? [y/N] ").strip().lower()
    if overwrite not in ('y', 'yes'):
        print("Aborting to avoid overwriting model.")
        exit()

# ----- Load dataset -----
print(f"Loading dataset: {train_path}")
dataset = CodeDataset(train_path, block_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----- Benchmark setup -----
benchmark_data = {
    "dataset": train_path,
    "total_tokens": dataset.total_tokens,
    "block_size": block_size,
    "batch_size": batch_size,
    "epochs": epochs,
    "start_time": time.time(),
    "steps": []
}

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
latest_loss = None

# ----- Training loop -----
print("\nBeginning training...")
while step < max_iters:
    for x, y in loader:
        if step >= max_iters:
            break

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        latest_loss = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start_time

        if step % eval_interval == 0:
            steps_done = step - resume_step
            steps_remaining = max_iters - step
            steps_per_sec = steps_done / max(elapsed, 1e-8)
            eta = steps_remaining / steps_per_sec if steps_per_sec > 0 else float('inf')

            confidence = estimated_confidence(loss)
            print(f"Step {step:>6} | Loss: {loss.item():.4f} | ~{confidence:.1f}% confidence | ETA: {eta/60:.1f} min")

            benchmark_data["steps"].append({
                "step": step,
                "loss": loss.item(),
                "confidence": confidence,
                "elapsed": elapsed
            })

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

if not benchmark_data["steps"] and latest_loss is not None:
    elapsed = time.time() - start_time
    benchmark_data["steps"].append({
        "step": step,
        "loss": latest_loss.item(),
        "confidence": estimated_confidence(latest_loss),
        "elapsed": elapsed
    })

# ----- Final save -----
torch.save(model.state_dict(), final_model_path)
print(f"\nTraining complete! Final model saved as: {final_model_path}")

# ----- Write benchmark report -----
benchmark_data["end_time"] = time.time()
benchmark_data["duration_sec"] = benchmark_data["end_time"] - benchmark_data["start_time"]
benchmark_data["steps_recorded"] = len(benchmark_data["steps"])
with open(benchmark_file, 'w') as f:
    json.dump(benchmark_data, f, indent=2)

print(f"Benchmark saved to {benchmark_file}")
