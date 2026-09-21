"""Audit the Step 6 offline dashboard against its balanced CSV source."""
import csv
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from giftcard_reviews.balanced_evaluation import FIELDS, metrics, select_balanced_reviews
from giftcard_reviews.sentiment import THREE_CLASS_LABELS

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / 'dashboard/index.html'
CSV_PATH = ROOT / 'outputs/balanced_three_class.csv'
DATA_PATH = ROOT / 'data/Gift_Cards.jsonl'


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.data_parts = []
        self.in_data = False
        self.external = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        if tag == 'script' and attrs.get('id') == 'dashboard-data':
            self.in_data = True
        for name in ('src', 'href'):
            value = attrs.get(name, '')
            if value and not value.startswith(('#', 'data:')):
                self.external.append(value)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.data_parts.append(data)


def main():
    document = HTML_PATH.read_text(encoding='utf-8')
    parser = Parser()
    parser.feed(document)
    embedded = json.loads(''.join(parser.data_parts))
    with CSV_PATH.open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames or ()) == FIELDS
        source_rows = list(reader)
    source_rows.sort(key=lambda row: int(row['sample_id']))
    assert len(source_rows) == 150
    selected = select_balanced_reviews(DATA_PATH)
    assert [int(row['row_id']) for row in source_rows] == [id for id, _ in selected]
    for source, record in zip(source_rows, embedded):
        for field in FIELDS:
            expected = source[field]
            if field in ('sample_id', 'row_id', 'rating', 'helpful_vote'):
                expected = int(float(expected))
            elif field in ('correct', 'verified_purchase'):
                expected = expected == 'True'
            assert record[field] == expected, (source['sample_id'], field)
        assert record['actual_sentiment'] == (
            'POSITIVE' if record['rating'] >= 4 else
            'NEUTRAL' if record['rating'] == 3 else 'NEGATIVE'
        )
        assert record['correct'] == (record['actual_sentiment'] ==
                                     record['predicted_sentiment'])
    assert Counter(r['actual_sentiment'] for r in embedded) == dict.fromkeys(
        THREE_CLASS_LABELS, 50)
    result = metrics(embedded)
    assert f"{result['accuracy']:.2%}" in document
    assert f"{result['balanced_accuracy']:.2%}" in document
    for actual in THREE_CLASS_LABELS:
        assert f"{result['recalls'][actual]:.2%}" in document
        for predicted in THREE_CLASS_LABELS:
            assert (f'<strong>{result["matrix"][actual,predicted]}</strong>'
                    in document)
    ratings = Counter(row['rating'] for row in embedded)
    for rating in range(1, 6):
        count = ratings[rating]
        pattern = (rf'data-rating="{rating}" data-count="{count}".*?'
                   rf'id="rating-bar-{rating}".*?data-value="{count}" '
                   rf'style="width:{count/50:.2%}"')
        assert re.search(pattern, document, re.S), f'Rating {rating} chart differs from CSV'
        assert count > 0 and count / 50 * 100 > 0
    predicted = Counter(row['predicted_sentiment'] for row in embedded)
    for label in THREE_CLASS_LABELS:
        count = predicted[label]
        assert re.search(rf'data-class="{label}".*?data-series="predicted" '
                         rf'data-count="{count}".*?data-value="{count}" '
                         rf'style="width:{count/75:.2%}"', document, re.S)
        right = result['matrix'][label, label]
        wrong = 50 - right
        assert (f'data-class="{label}" data-correct="{right}" '
                f'data-incorrect="{wrong}"') in document
        assert f'data-value="{right}" style="width:{right*2:.2f}%"' in document
        assert f'data-value="{wrong}" style="width:{wrong*2:.2f}%"' in document
    assert not parser.external, parser.external
    assert not re.search(r'fetch\s*\(|XMLHttpRequest|WebSocket|EventSource', document)
    assert {'search', 'result-filter', 'actual-filter', 'predicted-filter',
            'rating-filter', 'clear', 'result-count', 'review-rows'} <= parser.ids
    assert len(re.findall(r'<script>([\s\S]*?)</script>', document)) == 1
    print('PASS: 150 embedded rows exactly match the seeded source sample and CSV')
    print('PASS: three-class labels, accuracy, macro recall, and confusion cells match CSV')
    print('PASS: rating, actual/predicted, and correct/error chart values match CSV')
    print('PASS: explorer controls exist; dashboard has no external assets or network APIs')


if __name__ == '__main__':
    main()
