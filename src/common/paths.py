from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # jump back to root/
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"