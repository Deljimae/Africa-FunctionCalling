"""CPU regression checks, not GPU inference or full BFCL integration."""
import copy
from pathlib import Path
import unittest

from check_bfcl_core import parse_strict_json, checker_namespace, load_records
from generation_boundary import generation_settings, response_boundary

ROOT = Path(__file__).resolve().parents[1]


class BoundaryTests(unittest.TestCase):
    def test_numeric_overflow(self):
        for number in ['1e400', '-1e400', 'NaN', 'Infinity', '-Infinity']:
            for value in [number, '[' + number + ']', '{"nested":[' + number + ']}']:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    parse_strict_json('{"name":"test","arguments":{"value":' + value + '}}')

    def test_finite_numbers(self):
        for number in ['1e308', '-1e308', '0', '42']:
            self.assertTrue(parse_strict_json('{"name":"test","arguments":{"value":' + number + '}}'))

    def test_eos_at_budget(self):
        result = response_boundary([1, 2, 99], [99], 3)
        self.assertEqual(result['stop_reason'], 'eos')
        self.assertTrue(result['length_at_budget'])
        self.assertFalse(result['token_cap_reached'])
        self.assertEqual(result['content_token_ids'], [1, 2])

    def test_budget_without_eos(self):
        result = response_boundary([1, 2, 3], [99], 3)
        self.assertEqual(result['stop_reason'], 'max_new_tokens')
        self.assertTrue(result['token_cap_reached'])

    def test_only_suffix_removed(self):
        ids = [99, 1, 98, 2, 99, 100]
        result = response_boundary(ids, [99, 100], 20)
        self.assertEqual(result['content_token_ids'], [99, 1, 98, 2])
        self.assertEqual(ids, [99, 1, 98, 2, 99, 100])
        self.assertEqual(result['removed_terminal_token_ids'], [99, 100])

    def test_unknown_termination(self):
        self.assertEqual(response_boundary([1], [], 4)['stop_reason'], 'unknown')
        self.assertEqual(response_boundary([], [99], 4)['stop_reason'], 'unknown')

    def test_contract_budget_consumed(self):
        contract = {'generation': {'do_sample': False, 'max_new_tokens': 157}}
        self.assertEqual(generation_settings(contract)['max_new_tokens'], 157)
        for budget in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                generation_settings({'generation': {'do_sample': False, 'max_new_tokens': budget}})
        with self.assertRaises(ValueError):
            generation_settings({'generation': {'do_sample': True, 'max_new_tokens': 5}})

    def test_token_boundary_to_parser_to_checker(self):
        # Character IDs are a deterministic test codec, NOT a model tokenizer.
        folder = ROOT / 'audits/bfcl_core'
        ns = checker_namespace(folder / 'ast_checker.py')
        question = load_records(folder / 'questions.jsonl')['multiple_2']
        answer = load_records(folder / 'answers.jsonl')['multiple_2']
        raw = '{"name":"country_info.capital","arguments":{"country":"Brazil"}}'
        for suffix, accepted in [('', True), ('\nDone.', False), ('<|other|>', False)]:
            ids = list(map(ord, raw + suffix)) + [999999]
            boundary = response_boundary(ids, [999999], 200)
            text = ''.join(map(chr, boundary['content_token_ids']))
            if not accepted:
                with self.assertRaises(ValueError):
                    parse_strict_json(text)
                continue
            result = ns['ast_checker'](copy.deepcopy(question['function']), parse_strict_json(text),
                                      copy.deepcopy(answer['ground_truth']), ns['Language'].PYTHON,
                                      'multiple', 'fixture')
            self.assertTrue(result['valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
