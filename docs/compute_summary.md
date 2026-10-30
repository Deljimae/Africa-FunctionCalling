# Compute project summary

Updated 30 October 2026.

## What we are building

We are evaluating how African-language-focused models choose tools and supply
correct arguments in Yoruba, Hausa and Igbo. General-purpose models and English
requests provide comparisons. The study uses 200 public BFCL tasks, each with
multiple candidate tools and one expected call.

Translation generation is complete. We have distributed the translated tasks
to human reviewers, and reviewing is currently ongoing. Twelve tasks are reserved
for development, leaving 188 main-test candidates before further quality checks.

## Evidence of readiness

A 12-task English Llama-3.1-8B-Instruct development run completed on a Tesla T4
using 4-bit weights. All 12 outputs were valid JSON and selected the expected
function; 11 passed the complete-call check. Generation took 42.10 seconds.
Peak reserved CUDA allocator memory was 5.73 GiB. Model loading was measured
separately: 261.6 seconds initially and 69.3 seconds in the second process.
The restarted process reused two saved tasks and completed the remaining ten.

These are small development-run measurements, not representative multilingual
accuracy, full-device memory consumption or an estimate for all models.
The public runner includes tests for saved results and scoring. Recovery from
an actual mid-task GPU crash has not been demonstrated.

## Requested resources

- One GPU at a time, preferably with 48 GB memory.
- FP16/BF16 inference for models up to about 9B parameters.
- An initial setup and timing stage, followed by an agreed main-run allocation.
- Persistent working storage and downloadable outputs for our own backups.

The memory preference is not a measured minimum. Total GPU-hours will be refined
from measurements on the allocated hardware. The 4-bit T4 pilot does not establish
the timing or memory needed for FP16/BF16 on other models.

## Workload

For up to ten models, the planned comparison has seven conditions: English,
plus full and partial translation in each of three languages. At 188 candidate
tasks, one pass produces at most 1,316 task-condition outputs per model, or
13,160 across ten models. Final review exclusions and model-language coverage
may reduce that count. Development tests and any necessary reruns will be
budgeted separately. No model training is requested.

## Model candidates

General-purpose comparisons:
- meta-llama/Llama-3.1-8B-Instruct
- Qwen/Qwen3.5-9B

African-language-focused candidates:
- McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1
- NCAIR1/N-ATLaS
- Sunbird/Sunflower-Qwen3.5-9B
- CohereLabs/tiny-aya-earth
- Jacaranda/AfroLlama_V1
- llama-lang-adapt/AfriInstruct-Model with its required base model
- Lugha-Models/Lugha-Llama-8B-wura_edu
- vamboai/morena-1.5b-instruct

Final comparisons will document checkpoint settings and language coverage.

## When we can begin

Model setup and testing can begin once access is available. Main multilingual
evaluation depends on completed human review and correction of the translations.
Translation API funding is already covered. The workflow uses Python, PyTorch,
Hugging Face Transformers and Accelerate. A container has not yet been tested.

## Outputs and public benefit

We will save model responses, scores and run logs, with downloadable copies and
our own backup. We intend to publish evaluation code, reviewed translations where
licensing permits, and a technical report explaining the results and limitations.
These resources can help researchers and developers assess tool use in the
selected languages without repeating the same groundwork.

The study does not establish deployment safety or broad autonomous-agent capability.
