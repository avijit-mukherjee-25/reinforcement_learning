"""
Phase 3: PPO Optimization.

This is where all the RLHF machinery comes together. We instantiate four
models and let PPOTrainer run the rollout -> score -> GAE -> clipped update
loop.

Models:
    - policy        : trainable; starts from SFT
    - ref_policy    : frozen SFT, used for KL anchor
    - reward_model  : frozen, from Phase 2
    - value_model   : trainable; estimates V(s_t) for advantage computation

Per-token reward:
    r_t = -beta * log( pi_theta(y_t|.) / pi_ref(y_t|.) )       for t < T
    r_T = R_phi(x, y) - beta * log( pi_theta(y_T|.) / pi_ref(y_T|.) )

PPO update:
    L^CLIP = E[ min( r_t * A_hat, clip(r_t, 1-eps, 1+eps) * A_hat ) ]

PPOTrainer handles all of this internally; we only configure it.
"""

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer

# -- Config ---------------------------------------------------------------
SFT_MODEL = "./sft-model"
REWARD_MODEL = "./reward-model"
OUTPUT_DIR = "./ppo-model"

# -- Tokenizer ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -- Models ---------------------------------------------------------------
# Policy: trainable, starts from the SFT checkpoint.
policy = AutoModelForCausalLM.from_pretrained(SFT_MODEL)

# Reference policy: frozen copy of SFT, used purely for the KL term.
ref_policy = AutoModelForCausalLM.from_pretrained(SFT_MODEL)
for p in ref_policy.parameters():
    p.requires_grad = False

# Reward model: frozen, from Phase 2.
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL, num_labels=1
)
for p in reward_model.parameters():
    p.requires_grad = False

# Value model: trainable critic; in TRL's PPO, it is itself a sequence
# classifier (scalar head) initialized from the reward model so it has
# sensible features. The two are separate weight copies.
value_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL, num_labels=1
)

# Align pad ids on every model.
for m in (policy, ref_policy, reward_model, value_model):
    m.config.pad_token_id = tokenizer.pad_token_id

# -- Data: prompts only --------------------------------------------------
# PPO needs *only* prompts -- the policy rollouts produce the responses,
# which are then scored by the reward model. We reuse Anthropic/hh-rlhf
# prompts (extracting just the human turn). This keeps the prompt
# distribution aligned with what the reward model was trained on.
hh = load_dataset("Anthropic/hh-rlhf", split="train")


def extract_prompt(example):
    # Each "chosen" string starts with "\n\nHuman: ... \n\nAssistant: ..."
    # We take everything up to and including the first "Assistant:" tag.
    chosen = example["chosen"]
    cut = chosen.rfind("\n\nAssistant:")
    prompt = chosen[: cut + len("\n\nAssistant:")] if cut != -1 else chosen
    return {"prompt": prompt}


prompt_ds = hh.map(extract_prompt, remove_columns=hh.column_names)
prompt_ds = prompt_ds.shuffle(seed=42).select(range(20_000))


def tokenize(example):
    enc = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=512,
        padding=False,
    )
    return {"input_ids": enc["input_ids"], "lengths": len(enc["input_ids"])}


prompt_ds = prompt_ds.map(tokenize, remove_columns=["prompt"])
# Filter out prompts that got fully truncated to nothing.
prompt_ds = prompt_ds.filter(lambda ex: ex["lengths"] >= 8)

# -- PPO config -----------------------------------------------------------
config = PPOConfig(
    output_dir=OUTPUT_DIR,

    # Core PPO hyperparameters
    learning_rate=3e-6,
    num_ppo_epochs=4,             # K_ppo: minibatch passes per rollout
    num_mini_batches=1,
    cliprange=0.2,                # epsilon in clipped surrogate
    cliprange_value=0.2,
    vf_coef=0.1,                  # weight on value-function loss
    gamma=1.0,                    # no per-step discounting in LLM RLHF
    lam=0.95,                     # GAE lambda
    kl_coef=0.05,                 # beta -- KL penalty coefficient
    whiten_rewards=False,

    # Rollout / generation
    response_length=128,          # max generated tokens per response
    temperature=0.7,
    stop_token="eos",
    missing_eos_penalty=1.0,      # punish responses that never emit EOS

    # Total work
    total_episodes=10_000,        # episode = one prompt-response rollout
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    local_rollout_forward_batch_size=8,

    # Hardware / logging
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    num_sample_generations=10,    # periodic qualitative samples in logs
    save_strategy="steps",
    save_steps=500,
    report_to="none",
)

# -- Trainer --------------------------------------------------------------
trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=prompt_ds,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\n[Phase 3 done] PPO-aligned policy saved to {OUTPUT_DIR}")
