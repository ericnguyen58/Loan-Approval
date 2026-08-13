# Model Card

## Purpose
Estimates the likelihood a mortgage application would be approved at Truist,
Wells Fargo, or Bank of America, given an applicant's financial profile.

## Intended use
Educational/demo tool showing applicants a directional estimate and the
factors driving it. Not a substitute for applying with a lender, and not
used for any actual credit decision.

## Data
Home Mortgage Disclosure Act (HMDA) loan/application-level records, filtered
to the three target lenders. See `data/README.md` for source and fields.

## Explainability
Per-prediction SHAP values, surfaced as plain-language reasons in the style
of ECOA/Regulation B adverse-action notices.

## Fair lending / regulatory considerations
- Protected-class fields (race, ethnicity, sex) present in HMDA are excluded
  from model features; used only for post-hoc disparate-impact checks.
- Document known proxy-variable risks (e.g. geography, income) here as they're found.

### Post-hoc disparate-impact check (run on the held-out test set, n=63,410)

Reproducible via `src/loan_approval/explainability/disparate_impact.py`. Race,
ethnicity, and sex are not model inputs; they're joined back in afterward,
purely to audit the model's outputs. Two things are measured per group:
adverse impact ratio (that group's predicted-approval rate vs. the
highest-approval group's -- below 0.8 is the standard EEOC "4/5ths rule"
flag), and false-negative rate (among people *actually* approved, how often
the model wrongly called them denied).

| Group | n | Actual approval | Predicted approval | False-negative rate | Adverse impact ratio |
|---|---|---|---|---|---|
| White | 44,756 | 73.3% | 72.4% | 11.0% | 0.946 |
| Asian | 2,936 | 66.5% | 70.8% | 6.7% | 0.925 |
| **Black or African American** | 6,051 | 52.7% | 54.8% | 13.1% | **0.715 — flagged** |
| **American Indian or Alaska Native** | 298 | 45.6% | 48.7% | 11.0% | **0.635 — flagged** |
| Not Hispanic or Latino | 51,065 | 71.4% | 70.8% | 11.0% | 0.957 |
| **Hispanic or Latino** | 3,359 | 54.5% | 57.7% | 10.7% | **0.779 — flagged** |
| Male | 19,429 | 65.6% | 64.2% | 13.8% | 0.803 |
| **Female** | 14,093 | 63.3% | 59.6% | 15.5% | **0.745 — flagged** |

**Findings:**
1. **The model closely tracks each group's historical approval rate** (predicted
   is within ~1-4pp of actual, every group) -- it is not introducing new
   disparity beyond what's already in the historical lending data. But it is
   also not correcting for it: every flagged group here was already below the
   4/5ths threshold in the *actual* historical outcomes, before the model saw
   any of it. The model reproduces this via proxy signal (income, loan
   amount, credit-score type, which bank/geography) despite race/ethnicity/sex
   never being training features.
2. **False-negative rate is elevated specifically for Black and Female
   applicants** relative to White/Male (13.1% vs. 11.0%; 15.5% vs. 13.8%) --
   among applicants who were genuinely approved, the model is more likely to
   guess "denied" for these groups. Combined with a *lower* false-positive
   rate for the same groups (see script output), the model's errors skew
   systematically stricter for lower-base-rate groups, not just noisier.
   This is a known statistical tension (calibrating to match a group's own
   rate and having equal error rates across groups generally can't both hold
   when base rates differ) -- but it's a real, measured pattern here, not a
   hypothetical.
3. **Not yet done:** intersectional breakdowns (e.g. Black + Female), and
   whether the gap survives after controlling for income/DTI (i.e. is it
   proxy-driven or does the disparity persist within income bands too).

**Conclusion:** this is a demo/educational tool (see Intended use, above),
but if it or a similar model were ever used for an actual credit decision,
these findings -- inherited-not-created historical disparity, plus a
stricter-error skew for Black and Female applicants -- would need real
mitigation (e.g. group-aware threshold adjustment, reweighting, or a fairness
constraint during training) before that would be defensible, not just
documentation.

## Limitations
- HMDA covers mortgage lending only, not other loan products.
- Historical approval patterns may encode past discriminatory practices;
  outputs describe correlation in the data, not a normative standard.
