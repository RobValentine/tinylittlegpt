import torch
import math
from tokenizers import Tokenizer
from model.transformer import GPT
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

import argparse

def estimated_confidence(loss):
    return 100 * math.exp(-loss)

def generate(model, input_ids, tokenizer, device, max_new_tokens=64, temperature=1.0, top_k=5, readable_output=True, stop_at_newline=False):
    model.eval()
    generated = input_ids[:]
    decoded_output = tokenizer.decode(generated)

    for step in range(max_new_tokens):
        input_tensor = torch.tensor([generated[-config["block_size"]:]], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_tensor)

        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

        print(f"\nStep {step + 1}:")
        print("  Top predictions:")
        for i in range(top_k):
            tok_id = topk_indices[0][i].item()
            tok_prob = topk_probs[0][i].item()
            token = tokenizer.decode([tok_id]) if readable_output else tokenizer.id_to_token(tok_id) or f"[{tok_id}]"
            print(f"    {token:<20} (id: {tok_id:<5}) — P={tok_prob:.4f}")

        sampled_id = torch.multinomial(probs, num_samples=1).item()
        sampled_token = tokenizer.decode([sampled_id]) if readable_output else tokenizer.id_to_token(sampled_id) or f"[{sampled_id}]"
        print(f"  Selected: {sampled_token} (id: {sampled_id})")

        generated.append(sampled_id)

        if stop_at_newline and '\n' in tokenizer.decode([sampled_id]):
            print("Stopping at newline.")
            break

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="", help="Text prompt")
    parser.add_argument("--tokens", type=int, default=64, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k filtering")
    parser.add_argument("--readable", action="store_true", help="Readable output")
    parser.add_argument("--stop-at-newline", action="store_true", help="Stop generation when newline is generated")
    parser.add_argument("--output-file", type=str, help="Optional output file to write result")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    prompt = args.prompt.strip()
    enc = tokenizer.encode(prompt)
    input_ids = enc.ids

    print("\nPrompt:", prompt)

    device = torch.device(config["device"])
    model = GPT(config)
    model.load_state_dict(torch.load(config["gen_model_path"], map_location=device))
    model.eval()
    model.to(device)

    output_ids = generate(
        model, input_ids, tokenizer, device,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        readable_output=args.readable,
        stop_at_newline=args.stop_at_newline
    )

    decoded = tokenizer.decode(output_ids)
    final_output = decoded.replace(" ", "") if args.readable else decoded

    import io

def auto_indent(code: str) -> str:
    lines = code.splitlines()
    indented_lines = []
    indent_level = 0
    indent_str = "    "  # 4 spaces

    for line in lines:
        stripped = line.strip()
        if not stripped:
            indented_lines.append("")
            continue

        if stripped.startswith(("return", "pass", "break", "continue", "raise")) and indent_level > 0:
            indent_level -= 1

        indented_lines.append(f"{indent_str * indent_level}{stripped}")

        if stripped.endswith(":"):
            indent_level += 1

    return "\n".join(indented_lines)

if args.readable:
    formatted_output = auto_indent(decoded.replace(" ", ""))
    print("\nAuto-indented Output:")
    print(formatted_output)
else:
    print("\nOutput:")
    print(decoded)

if args.output_file:
    with open(args.output_file, "w") as f:
        f.write(final_output)
    print(f"\nOutput written to {args.output_file}")
