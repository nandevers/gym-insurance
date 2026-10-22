# Reproducing software checks and the research protocol

Use the corrected development branch, create a virtual environment and install
`python -m pip install -e ".[research,test]"`. Run `python -m pytest` and
`python -m scripts.synthetic_smoke --output artifacts/unique-smoke-name`.
The smoke workflow generates explicitly synthetic records, trains six small DQNs,
evaluates both splits and produces a report. It must use a fresh output directory.

For an authorized, verified real cohort with unique claim_id, ISO claim_date and
positive finite amount columns:

```sh
python -m gym_insurance.benchmark freeze --config configs/psr-v1.json --output artifacts/new-study
python -m gym_insurance.benchmark prepare --input VERIFIED_COHORT.csv --output artifacts/new-study
python -m gym_insurance.benchmark controls --output artifacts/new-study
python -m gym_insurance.benchmark train --output artifacts/new-study
python -m gym_insurance.benchmark evaluate --split validation --output artifacts/new-study
python -m gym_insurance.benchmark evaluate --split test --output artifacts/new-study
python -m gym_insurance.benchmark report --output artifacts/new-study
```

The PSR column claim_date is only a policy-date cohort proxy, not observed claim
arrival. Amount is already paid indemnity. Keep original raw bytes, resource IDs,
acquisition dates, schemas, filtering decisions and digests. Current official
downloads have not yet been aligned to the historical snapshot; do not claim exact
historical reproduction merely because an updated resource has a similar filename.

Freeze before inspecting test outcomes. Code/config/data/checkpoint identities are
verified. Do not overwrite a frozen study or silently change an incomplete matrix.
Separate executed settings from narrative recollection and keep every table linked
to its saved run. Raw personal fields must not become public fixtures or issue text.
