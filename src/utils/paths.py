from pathlib import Path

# Data Dir
# |- raw
# |- processed

ROOT = Path(__file__).resolve().parent.parent.parent

# ทีนี้ Path อื่นๆ ก็จะถูกต้องตามไปด้วย
DATA_DIR = ROOT / "data"

# Raw data directory paths
# |_ equity
# |_ economic
# |_ futures
RAW_DIR = DATA_DIR / "raw"

EQUITY_DIR = RAW_DIR / "equity"
PRICE_DIR = EQUITY_DIR / "price"
FUNDAMENTAL_DIR = EQUITY_DIR / "fundamental"

ECONOMIC_DIR = RAW_DIR / "economic"
FUTURES_DIR = RAW_DIR / "futures"
OPTIONS_DIR = RAW_DIR / "options"


# Processed data directory paths
PROCESSED_DIR = DATA_DIR / "processed"

# ---
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
VISUALIZATION_DIR = RESULTS_DIR / "visualization"