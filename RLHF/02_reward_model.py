"""
Phase 2: Reward Model training.

We initialize a sequence-classification head on top of the SFT model
(stripping the LM head and replacing it with a single scalar projection)
and train it under the Bradley-Terry preference model on (prompt, chosen,
rejected) triples from Anthropic/hh-rlhf.

Loss (computed automatically by RewardTrainer):
    L = -log sigmoid(R(x, y_w) - R(x, y_l))
"""

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

# -- Config ---------------------------------------------------------------
SFT_MODEL = "./sft-model"
OUTPUT_DIR = "./reward-model"
DATASET = "Anthropic/hh-rlhf"

# -- Tokenizer ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -- Model: SFT backbone + scalar head -----------------------------------
# `num_labels=1` swaps in a single-scalar classification head.
model = AutoModelForSequenceClassification.from_pretrained(
    SFT_MODEL,
    num_labels=1,
)
# Critical: align pad token between tokenizer and model config.
model.config.pad_token_id = tokenizer.pad_token_id

# -- Data preparation -----------------------------------------------------
# Anthropic/hh-rlhf format is:
#   {"chosen": "<full conversation incl. preferred response>",
#    "rejected": "<full conversation incl. dispreferred response>"}
#
# RewardTrainer expects columns named exactly "chosen" and "rejected"
# (each a string or a list-of-messages). The HH dataset already uses these
# column names, so no renaming needed.
ds = load_dataset(DATASET)

# Subset for tractability; remove for a real run.
train_dataset = ds["train"].shuffle(seed=42).select(range(30_000))
eval_dataset = ds["test"].shuffle(seed=42).select(range(2_000))

# -- Trainer --------------------------------------------------------------
config = RewardConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-6,               # RM lr is ~10x lower than SFT lr
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    max_length=1024,
    center_rewards_coefficient=0.01,  # small L2 toward zero-mean rewards
                                       # (helps PPO stability later)
    report_to="none",
)

trainer = RewardTrainer(
    model=model,
    args=config,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Quick sanity check: print the validation accuracy.
    metrics = trainer.evaluate()
    print(f"\n[Phase 2 done] Reward model saved to {OUTPUT_DIR}")
    print(f"Eval accuracy on held-out preference pairs: "
          f"{metrics.get('eval_accuracy', 'n/a')}")
