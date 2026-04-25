"""
Phase 1: Supervised Fine-Tuning (SFT).

Fine-tunes a base model on instruction-response demonstrations. The output
of this phase (`./sft-model/`) is used in Phase 2 as the initialization for
the reward model, and in Phase 3 as both the policy initialization and
the frozen reference policy for KL regularization.

Dataset: `trl-lib/Capybara` -- a clean conversational dataset already in TRL's
preferred chat format (list of {role, content} messages).
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

# -- Config ---------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-0.5B"   # any HF causal LM works
OUTPUT_DIR = "./sft-model"
DATASET = "trl-lib/Capybara"

# -- Data -----------------------------------------------------------------
# SFTTrainer accepts the conversational format directly; we only need the
# `messages` column with the chat template applied automatically.
dataset = load_dataset(DATASET, split="train")

# Smoke-test: subset to make this finish in reasonable time.
# Remove `.select(...)` for a real run.
dataset = dataset.shuffle(seed=42).select(range(20_000))

# -- Tokenizer ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -- Trainer --------------------------------------------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size 16
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,                        # use fp16=True if no bf16 hardware
    gradient_checkpointing=True,
    logging_steps=20,
    save_strategy="epoch",
    max_length=2048,
    packing=False,                    # set True for throughput on long data
    report_to="none",                 # change to "wandb" if you have it
)

trainer = SFTTrainer(
    model=BASE_MODEL,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[Phase 1 done] SFT model saved to {OUTPUT_DIR}")
