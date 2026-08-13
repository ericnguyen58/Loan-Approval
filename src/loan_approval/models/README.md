### Training

- Split the data, then built a `ColumnTransformer`:
  - Category cols: `["lei", "loan_type", "loan_purpose", "occupancy_type", "applicant_credit_score_type", "co-applicant_credit_score_type", "interest_only_payment", "negative_amortization", "balloon_payment", "aus-1", "aus-2", "aus-3"]` → One-Hot Encoded
  - Numeric cols: `['loan_amount', 'loan_term', 'income', 'debt_to_income_ratio', 'num_aus_used']` → StandardScaler

- Tried 2 models: **LogisticRegression** (baseline) vs **RandomForestClassifier** (captures non-linear signal LR can't). Both use `class_weight="balanced"` since the target (approved/denied) is imbalanced.

- Tuned both with `GridSearchCV` / `RandomizedSearchCV`, scored on **AUC-ROC**, not accuracy.
  > In plain English: if the model blindly approved every loan, it'd still score ~70% accuracy — a meaningless number for a business that needs to know *who* to approve, not just *how many*. AUC-ROC instead measures how well the model separates approved from denied applicants, which is what actually matters for ranking risk.

### Result: Random Forest wins

| Model | AUC-ROC |
|---|---|
| **RandomForestClassifier** | **0.918** |
| LogisticRegression | (baseline, lower) |

At the default 0.5 cutoff, pooled across all 3 banks:

| | precision | recall | f1-score | support |
|---|---|---|---|---|
| 0 (denied) | 0.69 | 0.82 | 0.75 | 18,774 |
| 1 (approved) | 0.92 | 0.84 | 0.88 | 44,636 |

Confusion matrix: `[[15404, 3370], [7083, 37553]]`

**What we observed: ** the model leans toward approving. Its "approved" calls are trustworthy 92% of the time, but its "denied" calls are only right 69% of the time — meaning roughly 3 in 10 people it flags as likely-denied would actually have been approved. For a tool that's meant to inform, not gatekeep, that's the safer direction to err (worse to falsely discourage someone from applying than to give an overoptimistic read) — but it's a trade-off worth stating out loud, not a hidden bias.

### Problem: one model, three very different banks

Because the model is scored per lender (`lei`) too, a pattern showed up that the pooled numbers hide entirely:

| Bank | n (test) | AUC-ROC | Actual approval rate | Predicted @ 0.5 |
|---|---|---|---|---|
| Truist | 26,157 | 0.855 | 74.6% | 75.0% |
| Wells Fargo | 21,164 | 0.964 | 71.1% | 63.0% |
| Bank of America | 16,089 | 0.906 | 62.7% | 49.6% |

Two things jump out:

1. **Wells Fargo's decisions are the most predictable from these features** (AUC 0.96):  its approve/deny pattern lines up cleanly with income, DTI, loan type, etc. **Truist is the least predictable** (AUC 0.85) — its outcomes look more idiosyncratic, i.e. more of Truist's real decision likely rides on signals this dataset doesn't capture (or more case by case judgment).
2. **A single global 0.5 cutoff badly underpredicts approval odds at Wells Fargo and Bank of America**:  telling users they're ~8-13 points less likely to get approved there than history actually shows. Truist looks fine by coincidence, not because the model treats it specially.

For a product whose whole pitch is *"compare your odds across 3 banks,"* that second point is a real bug, not a rounding error: a flat threshold would systematically make Wells Fargo and BofA look stingier than they are, and nudge every user toward Truist for reasons that have nothing to do with their actual chances.

### Fix: calibrate the threshold per bank

Instead of one global cutoff, each bank gets its own: pick the score cutoff that makes the model's predicted approval rate match *that bank's own historical* approval rate (a quantile cutoff on out of fold, non leaked training probabilities, applied untouched to the test set).

| Bank | Threshold | Actual approval rate | Predicted @ tuned threshold |
|---|---|---|---|
| Truist | 0.502 | 74.6% | 74.8% |
| Wells Fargo | 0.333 | 71.1% | 71.0% |
| Bank of America | 0.353 | 62.7% | 62.7% |

Payoff isn't just the headline rate matching — the denied class recall improves too (Wells Fargo 73% → 82%, Bank of America 90% → 76% with precision jumping from 66% → 76%), i.e. per bank calibration makes the model more balanced, not just better aimed.

Thresholds are saved to `models/thresholds.joblib`, keyed by `lei`.

### Business takeaway

The product isn't really "predict approve/deny" in the abstract — it's *"help someone see, honestly, which of these 3 banks is relatively more or less likely to say yes, and why."* A single global threshold quietly breaks that promise: it would make Wells Fargo and Bank of America look worse than Truist purely as a scoring artifact, not because their underwriting is actually stricter. Per bank calibration is what turns a generic risk score into a fair 3 ways comparison — it's not a nice to have, it's the feature.

One more caveat worth keeping visible to end users (see `docs/model_card.md`): a bank's high historical approval rate reflects what it *has* approved, which can encode past bias as easily as genuine leniency. "More likely to approve" should read as "historically said yes more to applicants like you," not "this bank is better" or "this bank is fair."


