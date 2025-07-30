import os
import time
import torch
import random
import logging
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast

from model.transformer import GPT
from model.layers import GPTDataset
from config import config

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
torch.manual_seed(config["seed"])
random.seed(config["seed"])

# Model
model = GPT(config).to(device)

# Dataset
dataset = GPTDataset(config["dataset_path"], config["block_size"])
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
loss_fn = CrossEntropyLoss()
scaler = GradScaler()

# Training loop
global_step = 0
model.train()

start_time = time.time()
for epoch in range(config["epochs"]):
    logger.info(f"Starting epoch {epoch + 1}/{config['epochs']}")
    for batch in loader:
        input_ids, targets = batch
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        # Logging
        if global_step % config["eval_interval"] == 0:
            logger.info(f"Step {global_step} — Loss: {loss.item():.4f}")

        # Checkpointing
        if global_step % config["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"checkpoint_step{global_step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

# Save final model
torch.save(model.state_dict(), config["final_model_path"])
logger.info(f"Training completed. Final model saved to {config['final_model_path']}")

end_time = time.time()
logger.info(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
