# Africa FunctionCalling

Evaluating how reliably African-language-focused models choose tools and supply
the right arguments in Yoruba, Hausa and Igbo, compared with general-purpose models.

## Why this matters

Understanding a request is not enough for a tool-using AI system. It must also
choose the right function and provide the values that function needs. This study
examines where those steps succeed or fail when requests are made in three
African languages.

## Current progress

Updated 30 October 2026.

- The evaluation starts from 200 public BFCL tasks. Each presents a request and
  several possible tools, with one expected function call.
- Translation drafts are complete for Yoruba, Hausa and Igbo. We have distributed
  the translated tasks to human reviewers, and reviewing is currently ongoing.
  The full set contains 600 task-language pairs, each with full and partial
  translation versions. Original drafts are retained separately from corrections.
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

- **Full translation (FT):** translate eligible request text while preserving
  names, identifiers and literal content where required by the task.
- **Partial translation (PAR):** additionally retain designated English argument
  expressions to examine how argument language affects function-call correctness.

Independent human reviews check meaning, specificity, naturalness and technical
details. Final review also checks that each FT/PAR pair preserves the same task.
Reviewers' initial judgments are retained separately from consensus corrections.

We measure whether responses have the required format, choose the correct tool
and provide the expected arguments. Language-dependent errors receive separate
semantic analysis, so a reference mismatch is not automatically treated as a
failure to understand the request.

Twelve source tasks are reserved for development. The remaining 188 are
main-test candidates, subject to review. Public benchmark exposure during model
training remains a limitation.

## Models

General-purpose baselines are **Llama-3.1-8B-Instruct** and **Qwen3.5-9B**.
African-language-focused candidates are AfriqueQwen3.5, N-ATLaS, Sunflower,
Tiny Aya Earth, AfroLlama, AfriInstruct, Lugha-Llama and MORENA.
Final comparisons will document checkpoint versions, language coverage and
prompting differences. Cross-model results will be descriptive comparisons,
not proof that language adaptation alone caused a performance difference.

## Development results

The 12-task English Llama pilot produced 12 valid JSON responses, selected the
reference function on all 12 tasks, and passed the complete-call check on 11.
The remaining failure involved an argument value. Generation took 42.10 seconds
on a Tesla T4 using 4-bit weights; peak reserved CUDA allocator memory was
5.73 GiB. Loading time was measured separately.

These measurements verify the development workflow, not multilingual accuracy.
Main evaluation is planned in FP16/BF16 after hardware validation.

## Next steps

- Complete human review, resolve corrections and check final FT/PAR consistency.
- Finalize task inclusion and source/reference issue handling before evaluation.
- Validate model settings and run the multilingual comparison.
- Analyse errors and release reviewed data, results and reproduction materials.

## Code and checks

The repository includes the Llama development runner, source checks and
offline tests. Further components will be released as they are prepared for reuse.
Python 3 is sufficient for the offline checks; no GPU or API key is required.

```bash
python3 scripts/check_bfcl_core.py
python3 -m unittest discover -s scripts -p 'test_*.py'
```

The source check verifies the saved data hashes and tests selected scoring
behaviours. It is not a full run of the official BFCL package.

For the GPU development runner, see [running the pilot](docs/running_the_pilot.md).

| Location | Contents |
| --- | --- |
| `scripts/` | Development inference runner, scoring checks and tests |
| `data/development_v1/` | Twelve development tasks and recorded main-test exclusions |
| `audits/bfcl_core/` | Pinned BFCL questions, references, checker and source hashes |
| `docs/` | Run instructions and supporting project documentation |

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
