from pathlib import Path

# Data Dir
# |- raw
# |- processed

ROOT = Path(__file__).resolve().parent.parent.parent

# ทีนี้ Path อื่นๆ ก็จะถูกต้องตามไปด้วย
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# คุณอาจจะเพิ่ม path อื่นๆ ที่ต้องใช้บ่อยๆ ไว้เลยก็ได้
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


# Raw data directory paths
# |_ equity
# |_ economic
# |_ futures
EQUITY_DIR = RAW_DIR / "equity"
ECONOMIC_DIR = RAW_DIR / "economic"
FUTURES_DIR = RAW_DIR / "futures"