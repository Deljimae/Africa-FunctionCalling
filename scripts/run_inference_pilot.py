"""Resumable, development-only Llama inference. No training or API calls."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
import uuid

from check_bfcl_core import checker_namespace, parse_strict_json
from generation_boundary import response_boundary

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
INSTRUCTION = ('Select exactly one tool. Return only a JSON object with keys name and arguments. '
               'Use the values and types required by the tool descriptions. No code fences or prose.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_contract(folder, contract):
    path = folder / 'contract.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('Run contract changed. Use a new output directory.')
    else:
        if any(folder.glob('results/*.json')):
            raise ValueError('Results exist without a run contract.')
        atomic_json(path, contract)


def messages(task):
    # Deliberate field allowlist: reference answers never enter inference prompts.
    return [{'role': 'user', 'content': INSTRUCTION + '\nTOOLS:\n' +
             json.dumps(task['tools'], ensure_ascii=False) + '\nREQUEST:\n' + task['english_request']}]


def score(task, text, namespace):
    try:
        parsed = parse_strict_json(text)
    except (ValueError, RecursionError):
        return {'parse_valid': False, 'function_correct': False, 'complete_call_correct': False}
    correct = list(parsed[0]) == list(task['ground_truth'][0])
    result = namespace['ast_checker'](copy.deepcopy(task['tools']), parsed,
              copy.deepcopy(task['ground_truth']), namespace['Language'].PYTHON, 'multiple', 'fixture')
    return {'parse_valid': True, 'function_correct': correct,
            'complete_call_correct': bool(result['valid']), 'checker_result': result}


def completed(folder, ids):
    rows = []
    for path in sorted((folder / 'results').glob('*.json')):
        row = json.loads(path.read_text())
        if row.get('id') not in ids or path.stem != row['id'] or row.get('status') != 'complete':
            raise ValueError('Unexpected or incomplete result: ' + path.name)
        rows.append(row)
    return rows


def score_saved(folder, task, namespace, retry_errors=False):
    """Score durable evidence; never regenerate or modify a generation record."""
    evidence = folder / 'results' / (task['id'] + '.json')
    row = json.loads(evidence.read_text())
    path = folder / 'scores' / evidence.name
    provenance = {'generation_sha256': digest(evidence),
                  'scorer_sha256': digest(Path(__file__)),
                  'checker_sha256': digest(ROOT / 'audits/bfcl_core/ast_checker.py')}
    if path.exists():
        previous = json.loads(path.read_text())
        if all(previous.get(k) == v for k, v in provenance.items()):
            if previous['status'] == 'scored' or not retry_errors:
                return previous
        # Retain prior attempts when rescoring or changing evaluator versions.
        history = folder / 'score_history' / (task['id'] + '_' + uuid.uuid4().hex + '.json')
        atomic_json(history, previous)
    result = {'id': task['id'], **provenance}
    try:
        result.update(status='scored', scores=score(task, row['response_content'], namespace))
    except Exception as exc:
        result.update(status='evaluation_error', error_type=type(exc).__name__, scores=None)
    atomic_json(path, result)
    return result


def summarize(folder, tasks):
    rows = completed(folder, {t['id'] for t in tasks})
    scored, errors, pending = [], 0, 0
    for row in rows:
        path = folder / 'scores' / (row['id'] + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            if record['generation_sha256'] != digest(folder / 'results' / path.name):
                raise ValueError('Score refers to different generation evidence')
            if record['status'] == 'scored':
                scored.append(record['scores'])
            elif record['status'] == 'evaluation_error':
                errors += 1
            else:
                raise ValueError('Unknown scoring status')
        elif 'scores' in row:  # Read-only compatibility with the archived v1 pilot.
            scored.append(row['scores'])
        else:
            pending += 1
    times = [r['generation_seconds'] for r in rows]
    summary = {'purpose': 'Development pipeline/timing pilot; not a model ranking',
               'completed': len(rows), 'planned': len(tasks),
               'all_complete': len(rows) == len(tasks),
               'generation_seconds_total': sum(times),
               'generation_seconds_median': statistics.median(times) if times else None,
               'generation_seconds_max': max(times) if times else None,
               'generated_tokens': sum(r['output_tokens'] for r in rows),
               'peak_allocated_bytes': max((r['peak_allocated_bytes'] for r in rows), default=0),
               'peak_reserved_bytes': max((r['peak_reserved_bytes'] for r in rows), default=0),
               'token_cap_hits': sum(r['boundary']['token_cap_reached'] for r in rows),
               'correct_calls': sum(s['complete_call_correct'] for s in scored),
               'limits': 'Generation timing excludes model loading. Sessions record loading separately. '
                         'First-task warmup is included; no cross-hardware/precision extrapolation.'}
    if any('scores' not in r for r in rows) or (folder / 'scores').exists():
        summary.update(scored=len(scored), evaluation_errors=errors, scoring_pending=pending,
                       all_scored=len(scored) == len(tasks),
                       all_complete=len(rows) == len(tasks) and errors == 0 and pending == 0,
                       generation_complete=len(rows) == len(tasks))
    atomic_json(folder / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['4bit', 'fp16', 'bf16'])
    parser.add_argument('--rescore-only', action='store_true', help='CPU only; retry evaluator errors from saved generations')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-tasks', type=int, default=12, help='Maximum NEW tasks this invocation')
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_tasks <= 0:
        parser.error('Budgets must be positive')
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # OS lock releases on process death; a second process cannot race the same run.
    import fcntl
    lock = (folder / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = ROOT / 'audits/bfcl_core'
    pack = ROOT / 'data/development_v1'
    manifest = json.loads((source / 'manifest.json').read_text())
    pack_manifest = json.loads((pack / 'manifest.json').read_text())
    if digest(source / 'ast_checker.py') != manifest['files']['ast_checker.py']['sha256']:
        raise ValueError('Checker hash mismatch')
    if digest(pack / 'tasks.jsonl') != pack_manifest['files']['tasks.jsonl']:
        raise ValueError('Development tasks hash mismatch')
    tasks = [json.loads(s) for s in (pack / 'tasks.jsonl').read_text().splitlines()]
    namespace = checker_namespace(source / 'ast_checker.py')
    if args.rescore_only:
        contract = json.loads((folder / 'contract.json').read_text())
        if contract['task_sha256'] != digest(pack / 'tasks.jsonl'):
            raise ValueError('Rescoring task bundle differs from original run')
        saved = {r['id'] for r in completed(folder, {t['id'] for t in tasks})}
        for task in tasks:
            if task['id'] in saved:
                score_saved(folder, task, namespace, retry_errors=True)
        print(json.dumps(summarize(folder, tasks)))
        return
    if args.precision is None:
        parser.error('--precision is required for GPU inference')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; no silent CPU fallback.')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported on this GPU.')
    versions = {p: importlib.metadata.version(p) for p in
                ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'tokenizers', 'huggingface-hub']}
    contract = {'version': 2, 'model': MODEL, 'revision': REVISION,
                'precision': args.precision, 'max_new_tokens': args.max_new_tokens,
                'decoding': 'greedy diagnostic, not frozen research protocol',
                'versions': versions, 'gpu': torch.cuda.get_device_name(0),
                'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
                'cuda': torch.version.cuda, 'instruction': INSTRUCTION,
                'task_sha256': digest(pack / 'tasks.jsonl'),
                'checker_sha256': digest(source / 'ast_checker.py'),
                'code_sha256': {p: digest(ROOT / 'scripts' / p) for p in
                    ['run_inference_pilot.py', 'generation_boundary.py', 'check_bfcl_core.py']}}
    ensure_contract(folder, contract)
    previous = completed(folder, {t['id'] for t in tasks})
    done = {r['id'] for r in previous}
    for task in tasks:
        if task['id'] in done:
            score_saved(folder, task, namespace)
    pending = [t for t in tasks if t['id'] not in done][:args.max_tasks]
    print(f'Skipping {len(done)} saved tasks; running {len(pending)} new tasks.', flush=True)
    if not pending:
        print(json.dumps(summarize(folder, tasks)))
        return
    session = {'session_id': uuid.uuid4().hex, 'skipped_ids': sorted(done), 'new_ids': [],
               'status': 'loading', 'started_unix': time.time()}
    session_path = folder / 'sessions' / (session['session_id'] + '.json')
    atomic_json(session_path, session)
    try:
        free, total = torch.cuda.mem_get_info()
        needed = 7 if args.precision == '4bit' else 20
        if free < needed * 2**30:
            raise RuntimeError(f'Pilot requires at least {needed} GiB free; use 4bit on a T4. This is a conservative guard, not a measured requirement.')
        start = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=True, trust_remote_code=False)
        dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
        options = {'torch_dtype': dtype, 'device_map': {'': 0}, 'attn_implementation': 'sdpa'}
        if args.precision == '4bit':
            options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                    token=True, trust_remote_code=False, **options).eval()
        torch.cuda.synchronize()
        session['load_seconds'] = time.perf_counter() - start
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        if not eos or not all(isinstance(x, int) for x in eos):
            raise ValueError('Missing model EOS configuration')
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos, pad_token_id=tok.eos_token_id,
                    bos_token_id=tok.bos_token_id, use_cache=True)
        metadata = {'effective_generation_config': config.to_dict(),
                    'template_sha256': hashlib.sha256(tok.get_chat_template().encode()).hexdigest(),
                    'dtype': str(dtype), 'attention': 'sdpa', 'device_map': {'': 0},
                    'quantization': options['quantization_config'].to_dict() if args.precision == '4bit' else None}
        meta_path = folder / 'generation_metadata.json'
        if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
            raise ValueError('Effective generation configuration changed on resume')
        atomic_json(meta_path, metadata)
        session['status'] = 'running'
        atomic_json(session_path, session)
        for task in pending:
            session['active_task'] = task['id']
            atomic_json(session_path, session)
            prompt = messages(task)
            ids = tok.apply_chat_template(prompt, tokenize=True, return_dict=False,
                      add_generation_prompt=True, date_string='21 Oct 2026')
            if len(ids) + args.max_new_tokens > 8192:
                raise ValueError('Pilot context guard exceeded; no silent truncation')
            inputs = torch.tensor([ids], device='cuda')
            mask = torch.ones_like(inputs)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(input_ids=inputs, attention_mask=mask,
                            generation_config=config, return_dict_in_generate=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            output_ids = generated[0, len(ids):].tolist()
            boundary = response_boundary(output_ids, eos, args.max_new_tokens)
            content = tok.decode(boundary['content_token_ids'], skip_special_tokens=False)
            row = {'id': task['id'], 'status': 'complete', 'session_id': session['session_id'],
                   'messages': prompt, 'input_token_ids': ids, 'input_tokens': len(ids),
                   'generated_token_ids': output_ids, 'output_tokens': len(output_ids),
                   'raw_output': tok.decode(output_ids, skip_special_tokens=False),
                   'response_content': content, 'boundary': boundary,
                   'generation_seconds': elapsed,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            atomic_json(folder / 'results' / (task['id'] + '.json'), row)
            session['new_ids'].append(task['id'])
            atomic_json(session_path, session)
            score_saved(folder, task, namespace)
            summarize(folder, tasks)
            print(task['id'], f'{elapsed:.2f}s', boundary['stop_reason'], flush=True)
            del generated, inputs, mask
        session['status'] = 'invocation_complete'
    except BaseException as exc:
        session['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        session['error_type'] = type(exc).__name__
        raise
    finally:
        session['ended_unix'] = time.time()
        atomic_json(session_path, session)
    print(json.dumps(summarize(folder, tasks)), flush=True)


if __name__ == '__main__':
    main()
