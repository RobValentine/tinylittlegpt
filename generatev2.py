import torch
import math
from tokenizers import Tokenizer
from model.transformer import GPT
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config
import argparse
import re
from tokenizers.implementations import ByteLevelBPETokenizer


def estimated_confidence(loss):
    return 100 * math.exp(-loss)

def generate(model, input_ids, tokenizer, device, max_new_tokens=64, temperature=1.0, top_k=5, readable_output=True):
    model.eval()
    generated = input_ids[:]
    
    for step in range(max_new_tokens):
        input_tensor = torch.tensor([generated[-config["block_size"]:]], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
        logits = logits[:, -1, :]  # Focus on last token
        logits = torch.clamp(logits, min=-10, max=10)
        logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        for token_id in set(generated):
            if token_id < probs.shape[1]:
                probs[0, token_id] *= 0.6

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

    return generated

def postprocess_python(text):
    """Refined Python reformatter for readability."""
    text = re.sub(r"(?<!\s)(def|class)(?=\w)", r"\1 ", text)
    text = re.sub(r"(?<! ):(?!\")", ":", text)
    text = re.sub(r"(?<=\w)(\()", r" \1", text)
    text = re.sub(r"(?<=[a-zA-Z0-9])(?=[=+\-*/])", " ", text)
    text = re.sub(r"(?<=[=+\-*/])(?=[a-zA-Z0-9])", " ", text)
    text = re.sub(r"\\n", "\n", text)
    return text.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="", help="Text prompt")
    parser.add_argument("--tokens", type=int, default=64, help="Max new tokens")
    parser.add_argument("--readable", action="store_true", help="Show decoded tokens instead of raw tokenizer output")
    parser.add_argument("--postprocess", action="store_true", help="Apply Python postprocessing to output")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k filtering for sampling")
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer(
        config["vocab_path"],
        config["merges_path"]
    )

    prompt = args.prompt.strip()
    enc = tokenizer.encode(prompt)
    input_ids = enc.ids

    print("\n🕅 Prompt:", prompt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert tokenizer.get_vocab_size() == config["vocab_size"], \
    f"Tokenizer vocab size {tokenizer.get_vocab_size()} doesn't match model config vocab size {config['vocab_size']}"

    print("Tokenizer vocab size:", tokenizer.get_vocab_size())
    print("Model config vocab size:", config["vocab_size"])
    assert tokenizer.get_vocab_size() == config["vocab_size"]

    model = GPT(config)
    model.load_state_dict(torch.load(config["gen_model_path"], map_location=device))
    model.eval()
    model.to(device)

    print("\n📊 Embedding weight stats:")
    print("Mean:", model.token_embedding.weight.mean().item())
    print("Std:", model.token_embedding.weight.std().item())    

    state_dict = torch.load(config["gen_model_path"], map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    output_ids = generate(
        model, input_ids, tokenizer, device,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        readable_output=args.readable
    )

    decoded = tokenizer.decode(output_ids)
    final_output = decoded if args.readable else decoded

    if args.postprocess:
        final_output = postprocess_python(final_output)

    print("\nOutput:")
    print(final_output)
