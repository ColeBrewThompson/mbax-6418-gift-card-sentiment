import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from giftcard_reviews.balanced_evaluation import (
    actual_sentiment, evaluate, metrics, select_balanced_reviews,
)
from giftcard_reviews.sentiment import (
    ThreeClassEmotionClassifier, parse_three_class_response,
)


class BalancedTests(unittest.TestCase):
    def test_three_class_mapping_and_validation(self):
        self.assertEqual([actual_sentiment(r) for r in (1, 2, 3, 4, 5)],
                         ['NEGATIVE', 'NEGATIVE', 'NEUTRAL', 'POSITIVE', 'POSITIVE'])
        with self.assertRaises(ValueError):
            actual_sentiment(0)
        with self.assertRaises(ValueError):
            actual_sentiment(6)

    def test_request_contains_only_title_and_text(self):
        reply = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"sentiment":"NEUTRAL","primary_emotion":"TRUST"}'
        }}]}
        with patch('giftcard_reviews.sentiment.urlopen',
                   return_value=io.BytesIO(json.dumps(reply).encode())) as send:
            result = ThreeClassEmotionClassifier().classify('Mixed', 'It was okay.')
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(json.loads(payload['messages'][1]['content']),
                         {'title': 'Mixed', 'text': 'It was okay.'})
        self.assertEqual(set(payload['response_format']['json_schema']['schema']
                             ['properties']['sentiment']['enum']),
                         {'POSITIVE', 'NEUTRAL', 'NEGATIVE'})
        self.assertEqual(result['sentiment'], 'NEUTRAL')
        with self.assertRaises(ValueError):
            parse_three_class_response('{"sentiment":"MIXED","primary_emotion":"JOY"}')

    def test_deterministic_sample_resume_and_metrics(self):
        reviews = [
            {'title': f'Review {i}', 'text': f'Body {i}',
             'rating': float((i % 5) + 1), 'verified_purchase': True,
             'helpful_vote': 0}
            for i in range(30)
        ]

        class FakeClassifier:
            def __init__(self):
                self.calls = []

            def classify(self, title, text):
                self.calls.append((title, text))
                return {'sentiment': 'NEUTRAL', 'primary_emotion': 'TRUST'}

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'reviews.jsonl'
            output = Path(directory) / 'balanced.csv'
            data.write_text(''.join(json.dumps(r) + '\n' for r in reviews),
                            encoding='utf-8')
            first = select_balanced_reviews(data, per_class=3, seed=17)
            self.assertEqual(first, select_balanced_reviews(data, per_class=3, seed=17))
            self.assertEqual(len(first), 9)
            self.assertEqual({actual_sentiment(r['rating']) for _, r in first},
                             {'POSITIVE', 'NEUTRAL', 'NEGATIVE'})
            classifier = FakeClassifier()
            rows = evaluate(data, output, 3, 17, classifier, workers=1)
            self.assertEqual(len(classifier.calls), 9)
            with output.open(newline='', encoding='utf-8') as source:
                saved = list(csv.DictReader(source))
            self.assertEqual(len(saved), 9)
            self.assertEqual([int(row['sample_id']) for row in saved],
                             list(range(1, 10)))
            classifier.calls.clear()
            self.assertEqual(evaluate(data, output, 3, 17, classifier, workers=1), rows)
            self.assertEqual(classifier.calls, [])
            summary = metrics(rows)
            self.assertEqual(summary['actual_counts'],
                             {'POSITIVE': 3, 'NEUTRAL': 3, 'NEGATIVE': 3})
            self.assertEqual(summary['recalls']['NEUTRAL'], 1)
            self.assertEqual(summary['balanced_accuracy'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
