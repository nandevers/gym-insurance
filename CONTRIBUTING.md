# Contributing

Check open issues and the roadmap before starting overlapping work. Use focused
branches and pull requests. Include the concrete problem, resulting behavior and
validation in the PR description. Link related issues without closing unfinished
scientific or licensing gates merely because a source patch is prepared.

Run the deterministic test suite and the synthetic workflow on a fresh directory.
Do not train against or tune on final test data for routine code changes. Changes
to protocol/package source require a new frozen study; historical outputs remain
immutable evidence of the version that generated them.

Share only synthetic minimal examples or permitted aggregate diagnostics. Do not
upload raw records, correspondence, reviewer reports, manuscripts, credentials or
private archives. See .gitignore and LICENSE-STATUS.txt. No new license grant or
contributor agreement is implied by this guide.
