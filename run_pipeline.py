import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    print(f"\n🔹 Running {script_path}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"❌ Failed at {script_path}")
        sys.exit(result.returncode)
    print(f"✅ Completed {script_path}")

if __name__ == "__main__":
    root = Path(__file__).parent

    steps = [
        "src/01_data_exploration.py",
        "src/02_data_cleaning.py",
        "src/03_train_test_split.py",
        "src/04_model_training.py",
        "src/05_feature_importance.py",
    ]

    print("🚀 Starting Restaurant Rating Prediction Pipeline")

    for step in steps:
        run_script(str(root / step))

    print("\n🎉 Pipeline completed successfully!")
