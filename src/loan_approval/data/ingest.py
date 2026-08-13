"""Load raw HMDA loan-level data for the target lenders (see data/README.md.md)."""
import requests

from loan_approval.config import DATA_RAW_DIR

BASE_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

# HMDA filer LEIs for the depository institutions themselves, not their
# holding companies (which don't file HMDA data).
BANK_LEIS = {
    "Truist Bank": "JJKC32MCHWDI71265Z06",
    "Wells Fargo Bank, N.A.": "KB1H1DSPRFMYMCUFXT09",
    "Bank of America, N.A.": "B4TYDEB6GKMZO031MB27",
}

# API rejects anything outside 2018-2023 (LEI-based reporting starts 2018;
# 2024+ isn't published yet).
YEARS = range(2018, 2024)


def fetch_hmda_csv() -> bytes:
    # The API silently truncates results to a single year whenever `years`
    # has more than one value combined with a `leis` filter (confirmed even
    # for a 3-year span) - so years must be fetched one request at a time
    # and concatenated, dropping the repeated header.
    chunks = []
    for year in YEARS:
        params = {"states": "NC", "years": str(year), "leis": ",".join(BANK_LEIS.values())}
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        lines = response.content.split(b"\n", 1)
        chunks.append(response.content if not chunks else lines[1])
    return b"\n".join(chunks)


if __name__ == "__main__":
    out_path = DATA_RAW_DIR / "hmda_data.csv"
    out_path.write_bytes(fetch_hmda_csv())
    print(f"Saved to {out_path}")
