import os
import time
import torch
import random
import logging
import sys
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
# Use CUDA AMP utilities for mixed precision training
from torch.cuda.amp import GradScaler, autocast

from model.transformer import GPT
from dataset import CodeDataset
from config import config

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
if config["seed"] is not None:
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    if config.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Model
model = GPT(config).to(device)

# Load latest good checkpoint
global_step = 0
resume_ckpt = None
checkpoint_dir = config["checkpoint_dir"]

# Scan and find latest good checkpoint (excluding known bad)
all_ckpts = sorted([
    f for f in os.listdir(checkpoint_dir)
    if f.startswith("checkpoint_step") and f.endswith(".pt")
], key=lambda x: int(x.split("checkpoint_step")[1].split(".pt")[0]))

for ckpt_file in reversed(all_ckpts):
    ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            global_step = int(ckpt_file.split("checkpoint_step")[1].split(".pt")[0])
            resume_ckpt = ckpt_path
            logger.info(f"🔄 Resumed from checkpoint: {ckpt_path}")
            break
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {ckpt_file}: {e}")
            continue

if not resume_ckpt:
    logger.info("🆕 Starting training from scratch.")

# Dataset
dataset = CodeDataset(config["train_path"], config["block_size"])
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
loss_fn = CrossEntropyLoss()
# Enable scaling only when running on CUDA
scaler = GradScaler(enabled=(device.type == "cuda"))

# Training loop
model.train()
start_time = time.time()

for epoch in range(config["epochs"]):
    logger.info(f"📚 Starting epoch {epoch + 1}/{config['epochs']}")
    for batch in loader:
        input_ids, targets = batch
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            if loss.item() > 20:
                logger.warning(f"⚠️ Loss spike at step {global_step} — skipping batch")
                continue

        if not torch.isfinite(loss):
            logger.error(f"❌ Non-finite loss at step {global_step}: {loss.item()}")
            bad_ckpt = os.path.join(checkpoint_dir, f"checkpoint_step{global_step}.pt")
            if os.path.exists(bad_ckpt):
                logger.warning(f"🧹 Removing corrupted checkpoint: {bad_ckpt}")
                os.remove(bad_ckpt)
            logger.info("⚠️ Stopping training to avoid model corruption.")
            sys.exit(1)  # <-- force exit instead of break

        scaler.scale(loss).backward()

        # NEW: Gradient clipping
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=0.25)

        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        # Logging
        if global_step % config["eval_interval"] == 0:
            logger.info(f"Step {global_step} — Loss: {loss.item():.4f}")

        # Save checkpoint
        if global_step % config["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step{global_step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"💾 Saved checkpoint to {ckpt_path}")

# Save final model
torch.save(model.state_dict(), config["final_model_path"])
logger.info(f"✅ Training completed. Final model saved to {config['final_model_path']}")

elapsed = time.time() - start_time
logger.info(f"⏱ Total training time: {elapsed/60:.2f} minutes")
