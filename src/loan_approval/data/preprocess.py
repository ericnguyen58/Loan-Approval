"""Clean and feature-engineer HMDA records into model-ready tables."""
import numpy as np
import pandas as pd

from loan_approval.config import DATA_PROCESSED_DIR

df = pd.read_csv(DATA_PROCESSED_DIR / "hmda_data_processed.csv", index_col=0)

# --- Drop target leakage ---
# `action_taken` is literally what `approved` was derived from. `denial_reason-1`
# is code 10 ("Not applicable") for every approved row and a real Reason code
# (1-9) for every denied row -- a perfect predictor of the target, not a real
# feature. `rate_spread`, `property_value`, and `loan_to_value_ratio` are only
# collected once a loan is priced/appraised -- i.e. after it's already approved,
# so their *missingness* alone predicts the target almost perfectly (e.g.
# rate_spread: 100% approved when present vs. 7% when missing). Verified
# directly against the data before dropping all of these.
leakage_cols = [
    "action_taken",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "rate_spread", "property_value", "loan_to_value_ratio","hoepa_status"
]
df = df.drop(columns=leakage_cols)

# `aus-4` is missing for every remaining row -- no signal, drop it.
df = df.drop(columns=["aus-4"])

# --- debt_to_income_ratio: convert bucketed strings to a numeric midpoint ---
# HMDA reports DTI as exact percentages in the 36%-49% band and coarse buckets
# outside it (e.g. "<20%", ">60%"); map buckets to their midpoint so the whole
# field becomes one ordinal numeric feature instead of loose category strings.
DTI_BUCKET_MIDPOINTS = {
    "<20%": 15,
    "20%-<30%": 25,
    "30%-<36%": 33,
    "50%-60%": 55,
    ">60%": 65,
}


def parse_dti(value):
    if pd.isna(value):
        return np.nan
    if value in DTI_BUCKET_MIDPOINTS:
        return DTI_BUCKET_MIDPOINTS[value]
    try:
        return float(value)
    except ValueError:
        return np.nan


df["debt_to_income_ratio"] = df["debt_to_income_ratio"].apply(parse_dti)

# --- Outlier capping ---
# A handful of sentinel/error values (loan_amount's $460M max) would otherwise
# dominate the scale for every real applicant. Cap at each column's own 99th
# percentile rather than a hardcoded number, so it stays correct if the
# underlying data changes.
outlier_cols = ["loan_amount", "income"]
for col in outlier_cols:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))
df["income"] = df["income"].clip(lower=0)  # negative income is a data error, not a real value

# --- Derived feature ---
# Number of automated underwriting systems used -- cheap signal from fields
# that are otherwise mostly missing (aus-2 is 86% missing, aus-3 94%).
df["num_aus_used"] = df[["aus-1", "aus-2", "aus-3"]].notna().sum(axis=1)

# --- Missing-value handling ---
numeric_cols = ["loan_amount", "loan_term", "income", "debt_to_income_ratio"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# aus-2/aus-3 missing means "no 2nd/3rd system reported" -- a real category,
# not noise, so it gets its own sentinel value rather than a median.
df[["aus-2", "aus-3"]] = df[["aus-2", "aus-3"]].fillna(-1).astype(int)

# Remaining categorical columns (lei, loan_type, loan_purpose, occupancy_type,
# hoepa_status, negative_amortization, interest_only_payment, balloon_payment,
# applicant_credit_score_type, co-applicant_credit_score_type, aus-1, aus-2,
# aus-3) are left as raw codes -- encoding is deferred to train.ipynb's pipeline
# so the encoder is fit on the train split only, not the full table.

out_path = DATA_PROCESSED_DIR / "hmda_features.csv"
df.to_csv(out_path)
print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to {out_path}")
