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

## Limitations
- HMDA covers mortgage lending only, not other loan products.
- Historical approval patterns may encode past discriminatory practices;
  outputs describe correlation in the data, not a normative standard.
