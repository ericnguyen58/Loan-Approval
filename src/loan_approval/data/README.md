### Summary works
## **What i have**
- # LAR Field Reference
    
    | Field | Description |
    |---|---|
    | lei | Financial institution's Legal Entity Identifier |
    | action_taken | Outcome of the loan application (originated, denied, withdrawn, etc.) |
    | debt_to_income_ratio | Applicant's total monthly debt as % of monthly income |
    | loan_to_value_ratio | Total debt secured by property vs. property value |
    | applicant_credit_score_type | Credit scoring model used for applicant |
    | co-applicant_credit_score_type | Credit scoring model used for co-applicant |
    | income | Gross annual income relied on in credit decision (in thousands) |
    | property_value | Value of property securing the loan |
    | loan_amount | Amount of the loan or amount applied for |
    | loan_type | Conventional, FHA, VA, or RHS/FSA loan |
    | occupancy_type | Principal residence, second residence, or investment property |
    | denial_reason-1 to -4 | Principal reason(s) for denial (up to 4) |
    | loan_term | Loan maturity term in months |
    | interest_only_payment | Whether loan includes interest-only payments |
    | negative_amortization | Whether loan terms allow negative amortization |
    | balloon_payment | Whether loan includes a balloon payment |
    | aus-1 to aus-5 | Automated underwriting system(s) used (up to 5) |
    | rate_spread | Difference between loan APR and average prime offer rate (APOR) |
    | hoepa_status | Whether loan is classified as high-cost (HOEPA) |
## **What I did**
- Mark all action_taken from value -> approved {0:denied,1:approved) -> DROP **action_taken**
- approved == 0 directly makes **denial_reason_1 -> 5** to be data leakeage -> DROP all **denied_reasons** columns
- Drop **aus4,5** just all null values. Keep **aus1 -> 3** and imputed missing with -1 (as unused AUS) instead of dropping or imputing with something else.
- Mask all DTI that have a range, use their mid-point 
  - Ex: _**20%-<30% -> 25**_
- **Income** has negative -> remove those records
- **"loan_to_value_ratio", "rate_spread", "property_value"** Dropped. Those columns
  only get collected once a loan is priced/appraised -- i.e. after it's already
  approved. So their *missingness* is directly data leakage on `approved`
  (verified: rate_spread present -> 100% approved, missing -> 7% approved).
- **Hoepa status** also a potential leakage (hoepa_status = 2 -> approved 100%)
> For plots, refer to : [preprocess.ipynb](preprocess.ipynb)
> For FE, refer to: [preprocess.py](preprocess.py)
> For saved figures, refer to: [figures](figures/)