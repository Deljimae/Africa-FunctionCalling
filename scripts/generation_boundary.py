"""Pure helpers shared by the self-contained smoke runner and CPU tests."""


def generation_settings(contract):
    settings = dict(contract['generation'])
    if set(settings) != {'do_sample', 'max_new_tokens'}:
        raise ValueError('Unsupported generation contract keys')
    if settings['do_sample'] is not False:
        raise ValueError('This diagnostic runner supports greedy decoding only')
    budget = settings['max_new_tokens']
    if type(budget) is not int or budget <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    return settings


def response_boundary(token_ids, eos_token_ids, max_new_tokens):
    """Remove only a suffix of explicitly configured EOS IDs, never text."""
    ids = list(token_ids)
    eos = set(eos_token_ids)
    end = len(ids)
    while end and ids[end - 1] in eos:
        end -= 1
    ended_eos = end < len(ids)
    at_budget = len(ids) >= max_new_tokens
    return {
        'content_token_ids': ids[:end],
        'removed_terminal_token_ids': ids[end:],
        'boundary_policy': 'configured_eos_suffix_only_v1',
        'eos_token_ids': sorted(eos),
        'length_at_budget': at_budget,
        'token_cap_reached': at_budget and not ended_eos,
        'stop_reason': 'eos' if ended_eos else ('max_new_tokens' if at_budget else 'unknown'),
    }
