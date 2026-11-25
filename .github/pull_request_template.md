## What changes for the user

<!-- One paragraph. Behaviour, not file names. -->

## Evidence

<!-- One row per claim. A claim without a row is not part of this pull request. -->

| Claim | Reached from (GUI control / CLI flag / caller) | Test that fails without it | Mutation tried |
|---|---|---|---|
| | | | |

## Not verified

<!-- Hardware that was only simulated, platforms not run, cases not covered. "Nothing" is an answer only if true. -->

## Checklist (see AGENTS.md)

- [ ] `pytest -q -rs` passes locally and every skip listed is a missing capability, not a wrong result
- [ ] `python tools/tests_bite.py --base <base branch>` passes; every `characterization` mark has a reason
- [ ] No new skip reason, `noqa`, `type: ignore`, loosened tolerance or widened `except`; if there is one, it is named here with its reason
- [ ] README, docs and docstrings say what the code does (no claim beyond the evidence table)
