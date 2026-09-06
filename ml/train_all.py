# ml/train_all.py
# Convenience script: runs all five training scripts back to back, so you
# can start it and walk away instead of running each command by hand.
import subprocess

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
        subprocess.run(["python", script], check=True)

    print("\nAll five models trained. Next: python evaluate.py, then python build_gate.py")
