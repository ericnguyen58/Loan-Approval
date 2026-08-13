# API — plain-English guide

This folder is the "front door" of the project: it's what turns the trained
model (built in `models/`) into something a website or app could actually
call. This doc walks through what was built, in plain language, no ML
background needed.

## The big picture

Think of it like a drive-through window for the model:

1. Someone sends in an applicant's info (income, loan amount, etc.) as a
   web request.
2. The code checks that the info actually makes sense (more on this below).
3. It runs that info through the trained model, once for each of the 3
   banks (Truist, Wells Fargo, Bank of America).
4. It sends back, for each bank: *how likely is this applicant to be
   approved, and would they be, yes or no.*

```
your request  →  [validation]  →  [the trained model]  →  ranked results
```

## The 3 files, in plain terms

| File | What it's for, in one sentence |
|---|---|
| `schemas.py` | Defines what a "valid applicant" looks like — the rulebook. |
| `routes.py` | Defines the actual web address (`/predict`) people send requests to. |
| `main.py` | Starts the whole app and decides what happens when something goes wrong. |

And one file outside this folder that this is all built on:

- `models/predict.py` — the part that actually knows how to load the
  trained model and turn "applicant info" into "a probability."

## What "validation" means here, and why it matters

Every field an applicant sends in has to make sense. A few examples of
rules now in place:

- `loan_type` can only be `1`, `2`, `3`, or `4` — those are the only 4 loan
  types that exist in the data (Conventional, FHA, VA, RHS/FSA). Sending
  `9` gets rejected.
- `debt_to_income_ratio` has to be between `0` and `100` — it's a
  percentage, it can't be `250`.
- `loan_amount` has to be a positive number, and capped at $5,000,000 —
  the model was never shown anything above that, so it has no business
  guessing about it.

**Why bother?** Two reasons:
1. **Garbage in, garbage out.** If someone accidentally sends `loan_type: 9`
   or a negative loan amount, the model would still spit out *some* number
   — but that number would be meaningless, and nothing would warn you.
2. **The model can only be trusted inside the range it learned from.**
   Values wildly outside what it was trained on aren't really predictions,
   they're guesses dressed up as predictions.

So instead of silently producing a bogus answer, the API now says "no,
that value doesn't make sense" and tells you exactly which field and why.

### What a rejected request looks like

If you send something invalid, you get back a clear list instead of one
big scary error dump:

```json
{
  "detail": [
    { "field": "loan_type", "issue": "Input should be 1, 2, 3 or 4" },
    { "field": "loan_amount", "issue": "Input should be greater than 0" },
    { "field": "debt_to_income_ratio", "issue": "Input should be less than or equal to 100" }
  ]
}
```

Each entry says exactly which field was wrong and why — no guessing.

## Try it yourself

Start the server:

```
uv run uvicorn loan_approval.api.main:app --reload
```

Then send it an applicant's info:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_type": 1, "loan_purpose": 1, "loan_amount": 250000, "loan_term": 360,
    "occupancy_type": 1, "income": 90, "debt_to_income_ratio": 33,
    "applicant_credit_score_type": 3, "co-applicant_credit_score_type": 10,
    "interest_only_payment": 2, "negative_amortization": 2, "balloon_payment": 2,
    "aus-1": 1, "aus-2": -1, "aus-3": -1, "num_aus_used": 1
  }'
```

You'll get back all 3 banks, ranked by how likely they are to approve
(`lei` is just each bank's official ID number, not something you need to
care about):

```json
[
  { "bank": "Wells Fargo Bank, N.A.", "lei": "KB1H...FXT09", "probability": 0.99, "threshold": 0.33, "approved": true },
  { "bank": "Truist Bank", "lei": "JJKC...1265Z06", "probability": 0.87, "threshold": 0.50, "approved": true },
  { "bank": "Bank of America, N.A.", "lei": "B4TY...MB27", "probability": 0.84, "threshold": 0.35, "approved": true }
]
```

`probability` is "how confident the model is," and `threshold` is the bar
that specific bank has to clear before we call it "approved" — each bank
has its *own* bar, calibrated to match how that bank has actually behaved
historically. (The full story on why every bank needs its own bar, instead
of one shared cutoff, is in `models/README.md` — short version: without
it, the tool would make some banks look stingier than they really are,
just because of a scoring quirk, not real underwriting differences.)

There's also a basic health check to confirm the server's alive:

```
curl http://localhost:8000/health
→ {"status": "ok"}
```

## Good to know

- This is a **demo/educational tool**, not a real loan decision — see
  `docs/model_card.md` for the fair-lending caveats.
- The list of fields the API expects (and their allowed values) matches
  the LAR field reference table in `data/README.md`.
