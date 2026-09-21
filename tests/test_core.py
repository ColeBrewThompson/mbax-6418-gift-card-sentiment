import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from giftcard_reviews.dataset import read_reviews
from giftcard_reviews.emotion_evaluation import (
    calculate_comparison,
    compare_sentiments,
    evaluate_emotions,
)
from giftcard_reviews.evaluation import calculate_metrics, evaluate
from giftcard_reviews.nrc_emotion import load_nrc_lexicon, score_emotions, tokenize
from giftcard_reviews.sentiment import (
    SentimentClassifier,
    SentimentEmotionClassifier,
    parse_sentiment,
    parse_sentiment_emotion,
)


def response():
    return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
        'content': '{"sentiment":"POSITIVE"}'}}]}).encode())


def emotion_response():
    return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
        'content': '{"sentiment":"POSITIVE","primary_emotion":"JOY"}'}}]}).encode())


class CoreTests(unittest.TestCase):
    def test_only_title_and_text_leave_machine(self):
        review = {'title': 'Nice', 'text': 'I enjoyed this.', 'rating': 1.0,
                  'verified_purchase': True, 'helpful_vote': 98765}
        with patch('giftcard_reviews.sentiment.urlopen', return_value=response()) as send:
            result = SentimentClassifier().classify(review['title'], review['text'])
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(json.loads(payload['messages'][1]['content']),
                         {'title': review['title'], 'text': review['text']})
        self.assertEqual(set(payload), {'model', 'temperature', 'max_tokens', 'messages', 'response_format'})
        self.assertEqual(result, 'POSITIVE')

    def test_invalid_results_rejected(self):
        for content in ['POSITIVE', '{"sentiment":"NEUTRAL"}', '{}',
                        '{"sentiment":"POSITIVE","reason":"good"}', 'null']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_sentiment(content)
        self.assertEqual(parse_sentiment('{"sentiment":"NEGATIVE"}'), 'NEGATIVE')

    def test_sentiment_emotion_response_and_request_boundary(self):
        review = {'title': 'Nice', 'text': 'I enjoyed this.', 'rating': 1.0,
                  'actual_sentiment': 'NEGATIVE', 'nrc_joy': 4}
        with patch('giftcard_reviews.sentiment.urlopen',
                   return_value=emotion_response()) as send:
            result = SentimentEmotionClassifier().classify(
                review['title'], review['text']
            )
        payload = json.loads(send.call_args.args[0].data)
        review_payload = json.loads(payload['messages'][1]['content'])
        self.assertEqual(review_payload, {'title': 'Nice', 'text': 'I enjoyed this.'})
        self.assertNotIn('rating', review_payload)
        self.assertNotIn('actual_sentiment', review_payload)
        self.assertNotIn('nrc_joy', review_payload)
        schema = payload['response_format']['json_schema']['schema']
        self.assertEqual(set(schema['required']), {'sentiment', 'primary_emotion'})
        self.assertEqual(result, {'sentiment': 'POSITIVE', 'primary_emotion': 'JOY'})

    def test_invalid_sentiment_emotion_results_rejected(self):
        invalid = [
            'POSITIVE',
            '{"sentiment":"POSITIVE"}',
            '{"sentiment":"POSITIVE","primary_emotion":"NEUTRAL"}',
            '{"sentiment":"POSITIVE","primary_emotion":"JOY","extra":1}',
        ]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_sentiment_emotion(content)

    def test_explicit_format_rejection_falls_back(self):
        error = HTTPError('url', 400, 'unsupported', {}, io.BytesIO(b'unsupported response_format'))
        with patch('giftcard_reviews.sentiment.urlopen', side_effect=[error, response()]):
            classifier = SentimentClassifier()
            self.assertEqual(classifier.classify('Nice', 'Good'), 'POSITIVE')
            self.assertEqual(classifier.response_mode, 'json_object')

    def test_missing_field_reports_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.jsonl'
            path.write_text('{"title":"hello"}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, "line 1: missing field 'text'"):
                list(read_reviews(path))

    def test_evaluation_is_resumable_and_only_passes_text_fields(self):
        reviews = [
            {'title': 'Good', 'text': 'Worked', 'rating': 5.0,
             'verified_purchase': True, 'helpful_vote': 2},
            {'title': 'Bad', 'text': 'Failed', 'rating': 1.0,
             'verified_purchase': False, 'helpful_vote': 0},
        ]

        class FakeClassifier:
            def __init__(self):
                self.calls = []

            def classify(self, title, text):
                self.calls.append((title, text))
                return 'POSITIVE' if title == 'Good' else 'NEGATIVE'

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / 'reviews.jsonl'
            output_path = Path(directory) / 'predictions.csv'
            with data_path.open('w', encoding='utf-8') as destination:
                for review in reviews:
                    destination.write(json.dumps(review) + '\n')
            classifier = FakeClassifier()
            first = evaluate(data_path, output_path, 2, classifier)
            self.assertEqual(classifier.calls, [('Good', 'Worked'), ('Bad', 'Failed')])
            classifier.calls.clear()
            resumed = evaluate(data_path, output_path, 2, classifier)
            self.assertEqual(classifier.calls, [])
            self.assertEqual(first, resumed)

    def test_metrics(self):
        rows = [
            {'actual_sentiment': 'POSITIVE', 'predicted_sentiment': 'POSITIVE'},
            {'actual_sentiment': 'POSITIVE', 'predicted_sentiment': 'NEGATIVE'},
            {'actual_sentiment': 'NEGATIVE', 'predicted_sentiment': 'NEGATIVE'},
            {'actual_sentiment': 'NEGATIVE', 'predicted_sentiment': 'POSITIVE'},
        ]
        metrics = calculate_metrics(rows)
        self.assertEqual(metrics['true_positives'], 1)
        self.assertEqual(metrics['true_negatives'], 1)
        self.assertEqual(metrics['false_positives'], 1)
        self.assertEqual(metrics['false_negatives'], 1)
        self.assertEqual(metrics['balanced_accuracy'], 0.5)

    def test_nrc_scoring_ties_and_no_match(self):
        lexicon = {
            'gift': {'anticipation', 'joy', 'trust'},
            'love': {'joy'},
            'afraid': {'fear'},
        }
        self.assertEqual(tokenize('GIFT!', 'I &amp; LOVE it.'),
                         ['gift', 'i', 'love', 'it'])
        tied = score_emotions('Gift', '', lexicon)
        self.assertEqual(tied['emotion'], 'ANTICIPATION')
        self.assertTrue(tied['tie'])
        self.assertEqual(tied['tied_emotions'], ('ANTICIPATION', 'JOY', 'TRUST'))
        self.assertEqual(tied['scores']['anticipation'], 1)
        no_match = score_emotions('xyz', '123', lexicon)
        self.assertEqual(no_match['emotion'], 'NO_MATCH')
        self.assertTrue(no_match['no_match'])

    def test_emotion_evaluation_resume_and_comparison(self):
        reviews = [
            {'title': 'Good', 'text': 'Love this gift', 'rating': 5.0,
             'verified_purchase': True, 'helpful_vote': 2},
            {'title': 'Bad', 'text': 'Unknown wording', 'rating': 1.0,
             'verified_purchase': False, 'helpful_vote': 0},
        ]

        class FakeClassifier:
            def __init__(self):
                self.calls = []

            def classify(self, title, text):
                self.calls.append((title, text))
                if title == 'Good':
                    return {'sentiment': 'POSITIVE', 'primary_emotion': 'JOY'}
                return {'sentiment': 'NEGATIVE', 'primary_emotion': 'ANGER'}

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data_path = directory / 'reviews.jsonl'
            output_path = directory / 'emotions.csv'
            original_path = directory / 'original.csv'
            lexicon_path = directory / 'nrc.txt'
            with data_path.open('w', encoding='utf-8') as destination:
                for review in reviews:
                    destination.write(json.dumps(review) + '\n')
            lexicon_path.write_text(
                'love\tjoy\t1\n'
                'gift\tjoy\t1\n'
                'gift\ttrust\t1\n',
                encoding='utf-8',
            )
            with original_path.open('w', newline='', encoding='utf-8') as destination:
                writer = csv.DictWriter(destination, fieldnames=(
                    'row_id', 'title', 'text', 'predicted_sentiment'
                ))
                writer.writeheader()
                writer.writerow({'row_id': 1, 'title': 'Good',
                                 'text': 'Love this gift',
                                 'predicted_sentiment': 'POSITIVE'})
                writer.writerow({'row_id': 2, 'title': 'Bad',
                                 'text': 'Unknown wording',
                                 'predicted_sentiment': 'NEGATIVE'})
            classifier = FakeClassifier()
            rows = evaluate_emotions(
                data_path, output_path, lexicon_path, 2, classifier
            )
            self.assertEqual(classifier.calls,
                             [('Good', 'Love this gift'), ('Bad', 'Unknown wording')])
            classifier.calls.clear()
            resumed = evaluate_emotions(
                data_path, output_path, lexicon_path, 2, classifier
            )
            self.assertEqual(classifier.calls, [])
            self.assertEqual(rows, resumed)
            self.assertEqual(compare_sentiments(rows, original_path), [])
            comparison = calculate_comparison(rows)
            self.assertEqual(comparison['comparable'], 1)
            self.assertEqual(comparison['agreeing'], 1)
            self.assertEqual(comparison['no_match'], 1)


if __name__ == '__main__':
    unittest.main()
