"""Shared paths and constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "src" / "loan_approval" / "data" / "figures"

# Target lenders, matched against HMDA `lei` / `respondent_name` fields during ingest.
BANKS = ["Truist", "Wells Fargo", "Bank of America"]

BANK_LEIS = {
    "Truist Bank": "JJKC32MCHWDI71265Z06",
    "Wells Fargo Bank, N.A.": "KB1H1DSPRFMYMCUFXT09",
    "Bank of America, N.A.": "B4TYDEB6GKMZO031MB27",
}
