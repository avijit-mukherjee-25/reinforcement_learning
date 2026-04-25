# RLHF End-to-End Example with TRL

A minimal but complete RLHF pipeline using HuggingFace TRL. Demonstrates all three phases on real open-source data with a small model that fits on a single 24GB GPU.

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| 1 | `01_sft.py` | Supervised fine-tuning on instruction-response pairs |
| 2 | `02_reward_model.py` | Train reward model on preference pairs |
| 3 | `03_ppo.py` | PPO optimization against the reward model |

## Models and data

- **Base model:** `Qwen/Qwen2.5-0.5B` (chosen for fit-on-laptop-GPU; the pipeline is identical for 7B+).
- **SFT data:** `trl-lib/Capybara` — multi-turn instruction-following pairs.
- **Preference data:** `Anthropic/hh-rlhf` (helpful/harmless preference pairs, the canonical RLHF dataset).
- **PPO prompts:** prompts extracted from `Anthropic/hh-rlhf` (the same prompts the preference data was collected on, so the reward model is in-distribution).

## Setup

```bash
# Tested with TRL 0.29 (PPOTrainer is now in trl.experimental.ppo).
pip install --upgrade "trl>=0.29" "transformers>=4.46" "datasets>=3.0" \
                       "accelerate>=1.0" "peft>=0.13" torch

# Optional but useful:
pip install wandb  # for training logs
```

## Run

```bash
# Phase 1 - takes ~20-40 min on a single A100/3090
python 01_sft.py

# Phase 2 - takes ~30-60 min
python 02_reward_model.py

# Phase 3 - takes a few hours; reduce --total_episodes for a quick smoke test
python 03_ppo.py
```

Or run all three with `python run_all.py`.

## What changed from older TRL examples

If you've seen older TRL tutorials, two things have changed:

1. The old `PPOTrainer.step()` loop is gone. The current `PPOTrainer` is a HuggingFace `Trainer` subclass — you build it with policy/ref/reward/value models and call `.train()`.
2. `PPOTrainer` is in `trl.experimental.ppo` (will be the only home from TRL 0.29+).

This example uses the current API.
