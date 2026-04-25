"""Run all three phases sequentially."""
import subprocess
import sys

scripts = ["01_sft.py", "02_reward_model.py", "03_ppo.py"]

for s in scripts:
    print(f"\n{'='*80}\nRunning {s}\n{'='*80}")
    rc = subprocess.call([sys.executable, s])
    if rc != 0:
        print(f"!! {s} failed with exit code {rc}")
        sys.exit(rc)

print("\n\nAll three phases complete. Run 04_compare.py to see SFT vs PPO outputs.")
