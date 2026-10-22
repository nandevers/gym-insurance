# Environment and protocol

`InsurEnv-v1` is the corrected development candidate on the v1 branch. The top-level
`gym_insurance.InsurEnv` alias selects v1; use explicit v0 for historical behavior.

An observation exposes the current amount and portfolio/budget state before the
action. Inclusion adds the amount exactly once. Primary potential reward is
distance-before minus distance-after; exclusion and infeasible inclusion pay zero.
An unaffordable record advances the stream so later affordable records remain
available. Decimal budget accounting prevents overspend; termination is explicit.

`configs/psr-v1.json` specifies 256-record windows contained within a policy year,
2006–2017 training, 2018–2019 validation and 2020–2021 test. Incomplete tails are
counted and excluded. Budgets and normalization use training data only. Ten seeds,
three budgets, two rewards and fixed 12,000-step DQN training were used locally.
No held-out tuning or checkpoint selection was performed.

Policies are reject-all, accept-if-feasible, budget-aware random, training-calibrated
amount threshold, myopic positive potential improvement and native PyTorch DQN.
Comparisons pair identical windows, then average within seed/year. Crossed seed/year
bootstrap intervals are conditional descriptions with only two observed test years.
Deterministic baseline repeats do not create extra independent evidence.

This is a first-digit total-variation proxy, not the historical three-digit statistic.
No fraud precision, recall, F1 or realized savings can be estimated without suitable
independent outcomes. Legitimate rounding/caps and label ambiguity require explicit
controls. Source-version reconciliation and external reproduction remain open in #2/#3.
