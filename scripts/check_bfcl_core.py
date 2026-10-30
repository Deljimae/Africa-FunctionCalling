"""Fetch a pinned research bundle and run bounded, offline CPU scoring fixtures."""
import argparse
import ast
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

REV = '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'
PREFIX = 'berkeley-function-call-leaderboard/bfcl_eval/'
FILES = {
    'questions.jsonl': PREFIX + 'data/BFCL_v4_multiple.json',
    'answers.jsonl': PREFIX + 'data/possible_answer/BFCL_v4_multiple.json',
    'ast_checker.py': PREFIX + 'eval_checker/ast_eval/ast_checker.py',
    'dataset_README.md': PREFIX + 'data/README.md',
    'LICENSE': 'LICENSE',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        raise RuntimeError('Existing manifest: use offline mode; never overwrite a snapshot.')
    entries = {}
    sources = {name: f'https://raw.githubusercontent.com/ShishirPatil/gorilla/{REV}/{path}'
               for name, path in FILES.items()}
    sources['hf_dataset_card.md'] = ('https://huggingface.co/datasets/gorilla-llm/'
                                    'Berkeley-Function-Calling-Leaderboard/raw/main/README.md')
    for name, url in sources.items():
        with urlopen(Request(url, headers={'User-Agent': 'research-source-audit'}), timeout=60) as response:
            data = response.read()
        (folder / name).write_bytes(data)
        entries[name] = {'url': url, 'sha256': digest(data), 'bytes': len(data)}
    (folder / 'manifest.json').write_text(json.dumps({
        'revision': REV, 'files': entries,
        'note': 'HF card is dated licensing evidence, not commit-pinned task data.',
        'retrieved_date': '2026-10-18'
    }, indent=2) + '\n', encoding='utf-8')


def load_records(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {row['id']: row for row in rows}
    assert len(rows) == len(result) == 200
    return result


def checker_namespace(path):
    # Preserve upstream function bodies/constants. Avoid loading model handlers,
    # provider SDKs or unsupported Java/JS dependencies for this Python-only test.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    namespace = {
        're': re,
        'Language': SimpleNamespace(PYTHON='python', JAVA='java', JAVASCRIPT='javascript'),
        'MODEL_CONFIG_MAPPING': {'fixture': SimpleNamespace(underscore_to_dot=False)},
    }
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def parse_strict_json(raw):
    """Proposed local parser, NOT the upstream BFCL response parser."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('Non-finite JSON number: ' + value)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    def check_finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Non-finite JSON number after conversion')
        if isinstance(item, dict):
            for child in item.values():
                check_finite(child)
        elif isinstance(item, list):
            for child in item:
                check_finite(child)
    check_finite(value)
    if not isinstance(value, dict) or set(value) != {'name', 'arguments'}:
        raise ValueError('Expected one name/arguments object')
    if not isinstance(value['name'], str) or not isinstance(value['arguments'], dict):
        raise ValueError('Invalid call fields')
    return [{value['name']: value['arguments']}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1] / 'audits/bfcl_core'
    if args.fetch:
        fetch(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['revision'] == REV
    for name, record in manifest['files'].items():
        assert digest((folder / name).read_bytes()) == record['sha256'], name
    questions = load_records(folder / 'questions.jsonl')
    answers = load_records(folder / 'answers.jsonl')
    assert questions.keys() == answers.keys()
    for key, answer in answers.items():
        assert len(answer['ground_truth']) == 1
        names = {f['name'] for f in questions[key]['function']}
        assert set(answer['ground_truth'][0]) <= names
    ns = checker_namespace(folder / 'ast_checker.py')
    fixtures = [
        ('correct', 2, [{'country_info.capital': {'country': 'Brazil'}}], True),
        ('normalized_string', 2, [{'country_info.capital': {'country': ' BRAZIL '}}], True),
        ('wrong_tool', 2, [{'country_info.population': {'country': 'Brazil'}}], False),
        ('wrong_value', 2, [{'country_info.capital': {'country': 'France'}}], False),
        ('missing_required', 2, [{'country_info.capital': {}}], False),
        ('extra_argument', 2, [{'country_info.capital': {'country': 'Brazil', 'extra': 1}}], False),
        ('wrong_type', 2, [{'country_info.capital': {'country': 42}}], False),
        ('optional_omitted', 0, [{'triangle_properties.get': {'side1': 5, 'side2': 4, 'side3': 3}}], True),
        ('accepted_alternative', 7, [{'wildlife_population.assess_growth': {'species': 'deer', 'location': 'WA', 'duration': 10}}], True),
        ('array_argument', 46, [{'sort_list': {'elements': ['Sam', 'Alice', 'Jack']}}], True),
        ('nested_dict', 9, [{'calculate_average': {'gradeDict': {'math': 90, 'science': 75, 'history': 82, 'music': 89}}}], True),
        ('extra_call', 2, [{'country_info.capital': {'country': 'Brazil'}}, {'country_info.capital': {'country': 'Brazil'}}], False),
    ]
    results = []
    for name, index, value, expected in fixtures:
        key = f'multiple_{index}'
        result = ns['ast_checker'](copy.deepcopy(questions[key]['function']), value,
                                   copy.deepcopy(answers[key]['ground_truth']),
                                   ns['Language'].PYTHON, 'multiple', 'fixture')
        results.append({'name': name, 'source_id': key, 'layer': 'upstream_checker_functions',
                        'expected': expected, 'actual': result['valid'], 'detail': result})
    good = '{"name":"country_info.capital","arguments":{"country":"Brazil"}}'
    for name, raw, expected in [
        ('plain_json', good, True), ('whitespace', '\n' + good + '\n', True),
        ('fenced_json', '```json\n' + good + '\n```', False),
        ('trailing_prose', good + '\nDone.', False),
        ('truncated', good[:-2], False), ('malformed', '{bad}', False),
        ('two_objects', good + good, False),
        ('duplicate_key', good.replace('"country":"Brazil"', '"country":"France","country":"Brazil"'), False),
    ]:
        try:
            parse_strict_json(raw)
            actual = True
        except (ValueError, TypeError):
            actual = False
        results.append({'name': name, 'layer': 'proposed_local_parser',
                        'expected': expected, 'actual': actual})
    report = {'purpose': 'CPU fixtures, not model results or full BFCL integration',
              'source_revision': REV, 'matched_records': len(questions),
              'limitations': ['Upstream function bodies extracted without imports.',
                             'Python only; identity function-name mapping fixture.',
                             'Strict JSON parser is a proposal, not approved main protocol.',
                             'No model inference, native handlers, or execution tested.'],
              'tests': results, 'passed': all(r['actual'] == r['expected'] for r in results)}
    (folder / 'scoring_fixture_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'tests': len(results), 'matched_records': len(questions)}))
    assert report['passed'], 'Inspect fixture report; do not change expectations to hide failures.'


if __name__ == '__main__':
    main()
