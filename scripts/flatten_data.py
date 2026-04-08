"""
Merges the Kaggle Training/ and Testing/ folders into
data/raw/<classname>/ so our dataset.py can re-split properly.
Run once: python scripts/flatten_data.py
"""
import shutil
from pathlib import Path

RAW = Path("data/raw")
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

for split in ["Training", "Testing"]:
    for cls in CLASSES:
        src = RAW / split / cls
        dst = RAW / cls
        if not src.exists():
            print(f"Skipping {src} — not found")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        files = list(src.glob("*"))
        for f in files:
            shutil.copy2(f, dst / f.name)
        print(f"Copied {len(files)} files: {src} → {dst}")

print("\nDone. You can delete data/raw/Training and data/raw/Testing now.")