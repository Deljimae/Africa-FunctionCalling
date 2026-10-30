# Africa FunctionCalling

Evaluating how reliably African-language-focused models choose tools and supply
the right arguments in Yoruba, Hausa and Igbo, compared with general-purpose models.

## Why this matters

Understanding a request is not enough for a tool-using AI system. It must also
choose the right function and provide the values that function needs. This study
examines where those steps succeed or fail when requests are made in three
African languages.

## Progress

Updated 30 October 2026.

- The evaluation starts from 200 public BFCL tasks. Each presents a request and
  several possible tools, with one expected function call.
- Translation drafts are complete for Yoruba, Hausa and Igbo. We have distributed
  the translated tasks to human reviewers, and reviewing is currently ongoing.
- A 12-task English development run with Llama-3.1-8B-Instruct completed on a
  Tesla T4 GPU. The system saved responses, scored calls and resumed across
  process restarts.
- Initial loading and text-generation checks have been completed for eight
  African-language-focused candidates.
- The main multilingual evaluation has not started. Loading checks and the
  small development run are not benchmark results or model rankings.

## Study design

We compare English requests with two translated versions: fully translated
requests and requests that retain selected English argument expressions.
Tool definitions and reference answers stay unchanged.

We measure whether responses have the required format, choose the correct tool
and provide the expected arguments. Language-dependent errors receive separate
semantic analysis, so a reference mismatch is not automatically treated as a
failure to understand the request.

Twelve source tasks are reserved for development. The remaining 188 are
main-test candidates, subject to review. Public benchmark exposure during model
training remains a limitation.

## Compute request

We seek GPU access for inference, not training. Our preference is one GPU with
48 GB memory, to run models up to about 9B parameters in FP16/BF16 rather than
relying on the 4-bit configuration used in the initial pilot.

We can begin model setup and testing while human review continues. Main
evaluation will begin after review and correction of the translations.
We propose measuring memory and runtime first, then agreeing the main-run budget.

[Read the compute project summary](docs/compute_summary.md)

## Code and checks

This public snapshot includes the Llama development runner, source checks and
offline tests. It is not yet a complete multi-model benchmark release.
Python 3 is sufficient for the offline checks; no GPU or API key is required.

```bash
python3 scripts/check_bfcl_core.py
python3 -m unittest discover -s scripts -p 'test_*.py'
```

The source check verifies the saved data hashes and tests selected scoring
behaviours. It is not a full run of the official BFCL package.

For the GPU development runner, see [running the pilot](docs/running_the_pilot.md).

## Planned outputs

We intend to release reviewed translations, evaluation configurations, results
and an error analysis, with source attribution and applicable licensing terms.
Translations are not included in this snapshot while review is ongoing.
No model weights, credentials or reviewer contact information are included.

## Source and attribution

Tasks, reference answers and scoring code come from
[Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard).
The saved revision and file hashes are in
[audits/bfcl_core/manifest.json](audits/bfcl_core/manifest.json).
The upstream [Apache 2.0 licence](audits/bfcl_core/LICENSE) is retained.
Third-party model terms apply separately.
