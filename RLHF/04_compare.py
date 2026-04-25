"""
Compare the SFT model against the PPO-aligned model on a few held-out prompts.
Run after Phase 3 finishes.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "How do I get better at public speaking?",
    "My friend is feeling down lately. What should I say to help?",
    "Can you explain why some people are afraid of flying?",
    "What's the safest way to teach a kid to ride a bike?",
]


def generate(model_path, prompt, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    formatted = f"\n\nHuman: {prompt}\n\nAssistant:"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("./sft-model")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for p in PROMPTS:
        print("=" * 80)
        print(f"PROMPT: {p}\n")
        print(f"-- SFT --\n{generate('./sft-model', p, tok)}\n")
        print(f"-- PPO --\n{generate('./ppo-model', p, tok)}\n")
