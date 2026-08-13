"""Post-hoc disparate-impact check.

Race, ethnicity, and sex are excluded from the model's training features (see
docs/model_card.md) -- but HMDA reports them, so they're reserved for exactly
this: checking whether the model's predictions or errors differ by group,
even though it was never shown group membership directly.

Two things are compared per group:
  - adverse_impact_ratio: that group's predicted-approval rate, divided by
    the highest-approval group's rate. Below 0.8 is the standard EEOC "4/5ths
    rule" threshold for flagging possible disparate impact.
  - false_negative_rate / false_positive_rate: among people who were actually
    approved/denied, how often the model got it wrong -- reveals whether the
    model's *mistakes*, not just its raw approval rate, land unevenly.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from loan_approval.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODEL_DIR

MIN_GROUP_SIZE = 200  # below this, a group's rate is too noisy to draw conclusions from
AIR_FLOOR = 0.8  # EEOC 4/5ths rule

# --- Reproduce the exact train/test split from train.ipynb ---
df = pd.read_csv(DATA_PROCESSED_DIR / "hmda_features.csv", index_col=0)
X = df.drop(columns=["approved"])
y = df["approved"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Pull protected-class fields back in by row position -- never used in training,
# only for this check. Same read pattern as preprocess.ipynb, so the row order (and
# therefore the index) lines up with hmda_features.csv without needing a join key. ---
demo_cols = ["derived_race", "derived_ethnicity", "derived_sex"]
demo = pd.read_csv(DATA_RAW_DIR / "hmda_data.csv", usecols=demo_cols).loc[X_test.index]

# --- Score the test set exactly as predict.py does: per-bank calibrated threshold ---
model = joblib.load(MODEL_DIR / "best_model.joblib")
thresholds = joblib.load(MODEL_DIR / "thresholds.joblib")
proba = model.predict_proba(X_test)[:, 1]
threshold = X_test["lei"].map(thresholds).to_numpy()
pred = (proba >= threshold).astype(int)

results = demo.copy()
results["actual"] = y_test.to_numpy()
results["predicted"] = pred


def summarize(group_col: str) -> pd.DataFrame:
    rows = []
    for group, g in results.groupby(group_col):
        if len(g) < MIN_GROUP_SIZE:
            continue
        actually_approved = g["actual"] == 1
        actually_denied = g["actual"] == 0
        rows.append({
            "group": group,
            "n": len(g),
            "actual_approval_rate": g["actual"].mean(),
            "predicted_approval_rate": g["predicted"].mean(),
            "false_negative_rate": (actually_approved & (g["predicted"] == 0)).sum() / actually_approved.sum(),
            "false_positive_rate": (actually_denied & (g["predicted"] == 1)).sum() / actually_denied.sum(),
        })
    out = pd.DataFrame(rows).sort_values("predicted_approval_rate", ascending=False)
    out["adverse_impact_ratio"] = out["predicted_approval_rate"] / out["predicted_approval_rate"].max()
    return out.reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    for col in demo_cols:
        table = summarize(col)
        flagged = table[table["adverse_impact_ratio"] < AIR_FLOOR]
        print("=" * 100)
        print(col, f"(groups with n < {MIN_GROUP_SIZE} dropped)")
        print("=" * 100)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if not flagged.empty:
            print(f"\n  FLAGGED (adverse impact ratio < {AIR_FLOOR}): {', '.join(flagged['group'])}")
        print()
