"""Stream and validate local reviews; metadata stays here."""
import json
from pathlib import Path

REQUIRED_TYPES = {
    'title': (str,), 'text': (str,), 'rating': (int, float),
    'verified_purchase': (bool,), 'helpful_vote': (int,),
}


def read_reviews(path):
    """Yield review dictionaries, raising a line-specific error for invalid data."""
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            try:
                review = json.loads(line)
                if not isinstance(review, dict):
                    raise ValueError('expected a JSON object')
                for field, types in REQUIRED_TYPES.items():
                    if field not in review:
                        raise ValueError(f'missing field {field!r}')
                    if type(review[field]) not in types:
                        raise ValueError(f'invalid type for {field!r}')
            except ValueError as exc:
                raise ValueError(f'{path}, line {line_number}: {exc}') from exc
            yield review
