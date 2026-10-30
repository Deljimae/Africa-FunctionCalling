# Running the development pilot

The included runner evaluates 12 English development tasks with
Llama-3.1-8B-Instruct. It is not the full multilingual evaluation.

## Offline verification

Run from the repository root with Python 3:

```bash
python3 scripts/check_bfcl_core.py
python3 -m unittest discover -s scripts -p 'test_*.py'
```

No network requests, GPU or credentials are needed for these commands. The source
check writes a local scoring-fixture report. Tests do not run model inference.

## GPU execution

Use Linux or WSL with a CUDA GPU and an environment containing PyTorch,
Transformers, Accelerate, bitsandbytes, tokenizers and huggingface-hub.
This snapshot does not provide a tested container or lockfile.
Access to the gated Llama checkpoint is required; authenticate securely outside
the repository. Never put access tokens in committed files.

```bash
python3 scripts/run_inference_pilot.py --output runs/llama_development --precision 4bit
```

The verified T4 run used 4-bit weights. For a larger compatible GPU, select
fp16 or bf16 in a new output directory. Those modes still require hardware
validation; do not treat the pilot measurements as their requirements.

Rerun the same command and configuration to reuse completed saved results.
Changed configurations require a new output directory. Do not delete saved
records to force a retry. CPU rescoring is available for an existing run:

```bash
python3 scripts/run_inference_pilot.py --output runs/llama_development --rescore-only
```

Each run records checkpoint revision, environment versions, responses, timing,
memory and scores. The 12 development IDs must remain outside the main test.
