# gym-insurance: versioned research implementation

`InsurEnv-v1` implements controlled sequential **distribution-proxy selection of historical paid indemnity records**.
It does not establish fraud detection accuracy, reduced fraud losses, or production suitability.
The November 2024 manuscript and historical experiments require separate scientific review.

## Versions and source preservation

The canonical development checkout is the repository root. Historical public
history remains on `main`; corrected development is on
`refactor/insurenv-v1-consolidation`. Complete pre-consolidation checkouts, local
changes, data, models and logs are preserved under ignored `.local/archive/`.
Git bundles and `refs/archive/` preserve branch history. Do not push archive refs,
private review materials or historical raw records. See `docs/CONSOLIDATION.md`.
The corrected source remains a local development candidate, not a public release.
Legacy environment files remain available under `gym_insurance/envs/`; v0 loads only
when requested and retains its historical optional dependencies and known limitations.
The new package entry point avoids importing v0 dependencies when using v1.
The top-level Python alias `gym_insurance.InsurEnv` now means v1 in this local
0.2 development candidate. For legacy code, use the explicit v0 registration or
the archived original snapshot package; this is not an unchanged top-level Python API.

The historical three-digit max-residual/L2 statistic is **not** the v1 metric.
The v1 metric is first-digit total variation: half the L1 difference between the
selected cohort's empirical frequencies and Benford probabilities. Empty selection
has explicit sentinel distance 1. This is a new, named protocol, not a corrected
reproduction of the paper's historical numbers.

## Environment contract

Input is a DataFrame with unique `claim_id`, ISO `claim_date`, and positive finite
`amount`. In this PSR experiment `claim_date` is a **policy-date cohort proxy**,
not a verified indemnity arrival timestamp. `VALOR_INDENIZACAO` is an amount already
**paid**, not a requested payout. Selection/include/exclude actions are retrospective
computational decisions, not prospective approval/refusal recommendations. No actual
claim-availability or settlement timestamp is present. No full-history engineered
personal features are used. The generic `claim_id` is a deduplicated proposal-record ID.

```python
import gymnasium as gym
import gym_insurance

env = gym.make("InsurEnv-v1", data=claims, budget=1000, amount_scale=100)
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(1)
```

The observation contains current amount/train scale, remaining budget/train scale,
remaining fraction, stream progress, selected count/stream size, nine selected-cohort
digit frequencies and nine current-digit indicators. No next claim or future amount
is exposed. Current amount is observable from the first decision. Input is copied.
`reset(seed=...)` uses a local RNG; optional shuffled synthetic tests are reproducible.
Real protocol windows retain chronological order.

The primary `potential_improvement` reward is distance before minus distance after.
Rejected and infeasible actions yield zero. Feasible approval computes its
counterfactual **before** mutating the selected cohort, adding the claim once.
Rewards telescope to 1 minus terminal distance. The explicitly named
`signed_counterfactual` ablation rewards a rejection with the negative feasible
approval delta; its cumulative reward does not have that telescoping interpretation.

Infeasible approvals reject and advance, so an expensive claim does not hide later
affordable claims. Decimal arithmetic prevents cent-level floating point overspend.
End of stream or zero remaining budget terminates. Only a configured horizon
truncates. Stepping after either flag raises an error. Resets/steps never write files.

## Frozen benchmark

`configs/psr-v1.json` specifies 256-claim windows wholly within a calendar year,
2006–2017 train / 2018–2019 validation / 2020–2021 test, and ten seeds. Incomplete
year tails are excluded and counted. Budgets are 10%, 25%, 50% of the **training-only**
median window total, held fixed for validation/test. Normalization and the amount
threshold use training-only medians. All policies see the same arriving windows.

Baselines are reject-all, accept-if-feasible, Bernoulli random, train-median amount
threshold and myopic positive potential improvement. The DQN implementation is
native CPU PyTorch with replay, a target network and fixed hyperparameters;
it is not Stable Baselines3. Gamma is 1 for the finite-horizon potential objective.
There is no validation/test hyperparameter tuning or checkpoint selection.

```text
python -m gym_insurance.benchmark freeze --config configs/psr-v1.json --output artifacts/psr-v1
python -m gym_insurance.benchmark prepare --input PATH_TO_VERIFIED_COHORT.csv --output artifacts/psr-v1
python -m gym_insurance.benchmark controls --output artifacts/psr-v1
python -m gym_insurance.benchmark train --output artifacts/psr-v1
python -m gym_insurance.benchmark evaluate --split validation --output artifacts/psr-v1
python -m gym_insurance.benchmark evaluate --split test --output artifacts/psr-v1
python -m gym_insurance.benchmark report --output artifacts/psr-v1
```

Freeze precedes opening the input. Protocol/code/data/checkpoint hashes are verified.
Changes require a fresh run directory; held-out evaluation cannot be overwritten.
Completed, matching training checkpoints can be resumed. Validation must finish
before test evaluation. Preparation rejects duplicate IDs and invalid amounts;
source reconciliation and exclusions belong in the separate provenance process.

Reports include terminal distance, claim/value coverage, budget utilization, runtime
and primary/ablation rewards. DQN-minus-myopic comparisons pair the same windows,
then average within seed/year cells. A crossed seed/year bootstrap gives a
conditional descriptive interval. Two test years provide weak evidence for temporal
generalization. Repeated deterministic baseline seeds do not create extra evidence.
No individual claim rows are treated as independent replicates for confidence intervals.

Prepared inputs and model files stay under ignored artifact directories. Reports
contain aggregate metrics and window identifiers, not personal claim identifiers.
Confirm data/software licensing, coauthor consent and manuscript state before release.

## Local runtime and checks

Normal environments can install this package with the `research,test` extras.
The Codex desktop's existing packages can also be read without changing shared
installations, using the launcher (bundled/scoped dependencies take precedence):

```text
python -s scripts/local_runtime.py -m pytest
python -s scripts/local_runtime.py -m scripts.synthetic_smoke --output artifacts/synthetic-smoke-v1
python -s scripts/local_runtime.py -m gym_insurance.benchmark --help
```

The smoke run generates clearly marked synthetic data and short real DQN training;
it verifies the pipeline, not the research hypothesis. Synthetic regression tests
check hand-calculated rewards, counterfactual inclusion, next observations, budgets,
seeds, no input mutation/file writes, split/scaling leakage, protocol integrity,
actual DQN updates/checkpoints and aggregate statistical units.

The separate frozen synthetic controls include a wholly legitimate rounded-amount
cohort, digit-independent synthetic labels, and opposite-label worlds with identical
observable amounts and identical policy decisions. These are illustrative checks of
what an unlabeled proxy can identify; they do not establish any real PSR fraud rate
or demonstrate that Benford is empirically invalid in that population.

## Install and reproduce software checks

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -e ".[research,test]"
python -m pytest
python -m scripts.synthetic_smoke --output artifacts/new-synthetic-smoke
```

Always choose a fresh smoke directory. Synthetic fits test the pipeline and do not
repeat or overwrite the frozen historical benchmark. Private review materials are
under `.local/research-review/`; the completed local study is under
`artifacts/psr-v1-local-20261022/`. Real data are not bundled for public distribution.

The fixed 12,000-step DQN performed worse than myopic selection across three
budgets under both rewards in the recorded historical study. Myopic record
coverage was only about 5.5–6.8%; a proxy advantage is not business superiority.
Only two held-out policy years support the conditional descriptive intervals.

## Release status

See `ROADMAP.md` and `LICENSE-STATUS.txt`. Source provenance, operative licensing,
human review and publication declarations remain open. Historical README badges
are not a verified license grant. No dataset, manuscript or package release is
authorized merely by this local consolidation.
