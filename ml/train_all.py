# ml/train_all.py
# runs all five training scripts back to back.
import subprocess
import sys

SCRIPTS = [
    "train_effnetv2s.py",
    "train_convnextv2.py",
    "train_swin.py",
    "train_vit.py",
    "train_dinov2_lora.py",
]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"\n{'='*20} Running {script} {'='*20}")
        # Use sys.executable to ensure it runs inside the active virtual environment
        subprocess.run([sys.executable, script], check=True)

    print("\nAll five models trained. Next: python evaluate.py, then python build_gate.py")