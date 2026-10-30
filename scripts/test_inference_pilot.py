"""CPU contract/recovery/scoring tests; never impersonates GPU measurements."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_inference_pilot import ROOT, atomic_json, completed, ensure_contract, messages, score, score_saved, summarize, digest
from check_bfcl_core import checker_namespace


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [json.loads(s) for s in
            (ROOT / 'data/development_v1/tasks.jsonl').read_text().splitlines()]
        self.ns = checker_namespace(ROOT / 'audits/bfcl_core/ast_checker.py')

    def test_reference_never_enters_prompt(self):
        task = copy.deepcopy(self.tasks[0])
        before = messages(task)
        task['ground_truth'] = 'SECRET_REFERENCE_SENTINEL'
        self.assertEqual(messages(task), before)
        self.assertNotIn('SECRET_REFERENCE_SENTINEL', json.dumps(messages(task)))

    def test_contract_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ensure_contract(folder, {'precision': '4bit'})
            ensure_contract(folder, {'precision': '4bit'})
            with self.assertRaises(ValueError):
                ensure_contract(folder, {'precision': 'fp16'})

    def test_recovery_keeps_wrong_outputs_and_ignores_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            row = {'id': self.tasks[0]['id'], 'status': 'complete', 'wrong': True}
            path = folder / 'results' / (row['id'] + '.json')
            atomic_json(path, row)
            path.with_suffix('.tmp').write_text('{')
            saved = completed(folder, {t['id'] for t in self.tasks})
            self.assertEqual(saved, [row])
            pending = [t for t in self.tasks if t['id'] not in {r['id'] for r in saved}]
            self.assertEqual(len(pending), 11)
            self.assertNotIn(row['id'], [t['id'] for t in pending])

    def test_failed_atomic_write_preserves_prior_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            atomic_json(path, {'original': True})
            with patch('run_inference_pilot.os.replace', side_effect=OSError('simulated interruption')):
                with self.assertRaises(OSError):
                    atomic_json(path, {'original': False})
            self.assertEqual(json.loads(path.read_text()), {'original': True})
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_corrupt_result_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            atomic_json(folder / 'results/bad.json', {'id': 'unknown', 'status': 'complete'})
            with self.assertRaises(ValueError):
                completed(folder, {t['id'] for t in self.tasks})

    def test_real_task_scoring(self):
        task = self.tasks[0]
        good = json.dumps({'name': 'corporate_finance.dividend_data',
                          'arguments': {'company': 'Apple', 'years': 5}})
        self.assertTrue(score(task, good, self.ns)['complete_call_correct'])
        bad = good.replace('"years": 5', '"years": 50')
        self.assertTrue(score(task, bad, self.ns)['function_correct'])
        self.assertFalse(score(task, bad, self.ns)['complete_call_correct'])
        self.assertFalse(score(task, good + '\nDone', self.ns)['parse_valid'])
        wrong_tool = json.dumps({'name': 'stock_market_data', 'arguments': {'company': 'Apple', 'days': 5}})
        self.assertFalse(score(task, wrong_tool, self.ns)['function_correct'])

    def overflow_evidence(self, folder):
        task = next(t for t in self.tasks if t['id'] == 'multiple_57')
        name, alternatives = next(iter(task['ground_truth'][0].items()))
        arguments = {k: v[0] for k, v in alternatives.items()}
        arguments['rate'] = 10**400
        text = json.dumps({'name': name, 'arguments': arguments})
        row = {'id': task['id'], 'status': 'complete', 'response_content': text,
               'generation_seconds': 1.0, 'output_tokens': 1,
               'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0,
               'boundary': {'token_cap_reached': False}}
        atomic_json(folder / 'results' / (task['id'] + '.json'), row)
        return task, text

    def test_overflow_preserves_evidence_and_is_not_wrong_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task, text = self.overflow_evidence(folder)
            with self.assertRaises(OverflowError):
                score(task, text, self.ns)
            path = folder / 'results' / (task['id'] + '.json')
            before = path.read_bytes()
            result = score_saved(folder, task, self.ns)
            self.assertEqual(result['status'], 'evaluation_error')
            self.assertEqual(result['error_type'], 'OverflowError')
            self.assertIsNone(result['scores'])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(completed(folder, {task['id']})), 1)
            summary = summarize(folder, [task])
            self.assertTrue(summary['generation_complete'])
            self.assertFalse(summary['all_complete'])
            self.assertEqual(summary['scored'], 0)
            self.assertEqual(summary['evaluation_errors'], 1)

    def test_interruption_between_generation_and_score_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task, _ = self.overflow_evidence(folder)
            with patch('run_inference_pilot.score', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    score_saved(folder, task, self.ns)
            self.assertEqual(summarize(folder, [task])['scoring_pending'], 1)
            result = score_saved(folder, task, self.ns)
            self.assertEqual(result['status'], 'evaluation_error')
            self.assertEqual(len(completed(folder, {task['id']})), 1)

    def test_offline_rescore_cli_needs_no_gpu(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task, _ = self.overflow_evidence(folder)
            atomic_json(folder / 'contract.json',
                        {'task_sha256': digest(ROOT / 'data/development_v1/tasks.jsonl')})
            command = [sys.executable, str(ROOT / 'scripts/run_inference_pilot.py'),
                       '--output', str(folder), '--rescore-only']
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout)['evaluation_errors'], 1)
            self.assertEqual(len(list((folder / 'score_history').glob('*.json'))), 1)
            self.assertEqual(len(completed(folder, {task['id']})), 1)


if __name__ == '__main__':
    unittest.main()
