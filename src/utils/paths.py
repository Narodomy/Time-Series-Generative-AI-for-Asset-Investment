import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# === SOURCE DATA ===
SOURCE_DIR = ROOT / "source"
STOCKS_DIR = SOURCE_DIR / "stocks"  # AAPL/, NVDA/, etc with price & fundamental
MACRO_DIR = SOURCE_DIR / "macro"  # Interest rates, indices, forex
REFERENCE_DIR = SOURCE_DIR / "reference"  # Metadata, ticker lists

# === PROCESSED DATA ===
FEATURES_DIR = SOURCE_DIR / "features"  # Engineered features
DATASETS_DIR = SOURCE_DIR / "datasets"  # Train/test splits

# === EXPERIMENTS ===
EXPERIMENTS_DIR = ROOT / "02_experiments"
PROCESSED_DIR = ROOT / "01_processed"

# === LEGACY (optional compatibility) ===
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CHECKPOINTS_DIR = ROOT / "checkpoints"

REPORTS_QS_DIR = ROOT / "report_qs"
