"""Resumable evaluation of the frozen sentiment classifier."""
import argparse
import csv
import os
import sys
import textwrap
import time
from itertools import islice
from pathlib import Path

from .dataset import read_reviews
from .sentiment import LABELS, SentimentClassifier


CSV_FIELDS = (
    'row_id',
    'title',
    'text',
    'rating',
    'actual_sentiment',
    'predicted_sentiment',
    'correct',
    'verified_purchase',
    'helpful_vote',
)


def actual_sentiment(rating):
    """Convert a local rating to the assignment's binary ground truth."""
    return 'POSITIVE' if rating >= 4 else 'NEGATIVE'


def first_reviews(path, limit=100):
    """Load exactly the first ``limit`` validated reviews."""
    reviews = list(islice(read_reviews(path), limit))
    if len(reviews) != limit:
        raise ValueError(f'Expected at least {limit} reviews; found {len(reviews)}')
    return reviews


def _expected_row(review, row_id, prediction):
    """Build local evaluation fields only after a prediction exists."""
    actual = actual_sentiment(review['rating'])
    return {
        'row_id': str(row_id),
        'title': review['title'],
        'text': review['text'],
        'rating': str(review['rating']),
        'actual_sentiment': actual,
        'predicted_sentiment': prediction,
        'correct': str(prediction == actual),
        'verified_purchase': str(review['verified_purchase']),
        'helpful_vote': str(review['helpful_vote']),
    }


def load_completed(output_path, reviews):
    """Load and validate resumable results against the current source rows."""
    output_path = Path(output_path)
    if not output_path.exists():
        return {}
    with output_path.open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f'{output_path} has an unexpected CSV header')
        completed = {}
        for csv_line, row in enumerate(reader, 2):
            try:
                row_id = int(row['row_id'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{output_path}, line {csv_line}: invalid row_id') from exc
            if row_id < 1 or row_id > len(reviews) or row_id in completed:
                raise ValueError(f'{output_path}, line {csv_line}: unexpected row_id {row_id}')
            prediction = row['predicted_sentiment']
            if prediction not in LABELS:
                raise ValueError(f'{output_path}, line {csv_line}: invalid prediction')
            expected = _expected_row(reviews[row_id - 1], row_id, prediction)
            if row != expected:
                raise ValueError(
                    f'{output_path}, line {csv_line}: saved row does not match source data'
                )
            completed[row_id] = row
    return completed


def append_prediction(output_path, row):
    """Append and fsync one completed result so an interruption loses no prior rows."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists()
    with output_path.open('a', newline='', encoding='utf-8') as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        destination.flush()
        os.fsync(destination.fileno())


def classify_with_retries(classifier, title, text, attempts=3):
    """Retry transient classifier failures without changing the frozen prompt."""
    for attempt in range(1, attempts + 1):
        try:
            return classifier.classify(title, text)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(attempt)


def evaluate(data_path, output_path, limit=100, classifier=None):
    """Evaluate the first rows, resuming any predictions already saved."""
    reviews = first_reviews(data_path, limit)
    completed = load_completed(output_path, reviews)
    classifier = classifier or SentimentClassifier()

    if completed:
        print(f'Resuming with {len(completed)} of {limit} predictions already saved.', flush=True)
    for row_id, review in enumerate(reviews, 1):
        if row_id in completed:
            continue

        # The external boundary receives only title and text. Ground truth is
        # intentionally constructed on the following line, after this returns.
        prediction = classify_with_retries(
            classifier,
            title=review['title'],
            text=review['text'],
        )
        row = _expected_row(review, row_id, prediction)
        append_prediction(output_path, row)
        completed[row_id] = row
        print(f'Completed {len(completed)}/{limit}: row {row_id}', flush=True)

    return [completed[row_id] for row_id in range(1, limit + 1)]


def calculate_metrics(rows):
    """Calculate assignment metrics with POSITIVE as the positive class."""
    tp = sum(r['actual_sentiment'] == 'POSITIVE' and
             r['predicted_sentiment'] == 'POSITIVE' for r in rows)
    tn = sum(r['actual_sentiment'] == 'NEGATIVE' and
             r['predicted_sentiment'] == 'NEGATIVE' for r in rows)
    fp = sum(r['actual_sentiment'] == 'NEGATIVE' and
             r['predicted_sentiment'] == 'POSITIVE' for r in rows)
    fn = sum(r['actual_sentiment'] == 'POSITIVE' and
             r['predicted_sentiment'] == 'NEGATIVE' for r in rows)
    actual_positive = tp + fn
    actual_negative = tn + fp
    positive_recall = tp / actual_positive if actual_positive else 0.0
    negative_recall = tn / actual_negative if actual_negative else 0.0
    total = len(rows)
    correct = tp + tn
    return {
        'total': total,
        'correct': correct,
        'incorrect': fp + fn,
        'accuracy': correct / total if total else 0.0,
        'actual_positive': actual_positive,
        'actual_negative': actual_negative,
        'positive_recall': positive_recall,
        'negative_recall': negative_recall,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'balanced_accuracy': (positive_recall + negative_recall) / 2,
    }


def print_report(rows):
    """Print metrics and a readable table of every incorrect prediction."""
    metrics = calculate_metrics(rows)
    print('\nEvaluation metrics')
    print(f"Total reviews: {metrics['total']}")
    print(f"Correct predictions: {metrics['correct']}")
    print(f"Incorrect predictions: {metrics['incorrect']}")
    print(f"Overall accuracy: {metrics['accuracy']:.2%}")
    print(f"Actual POSITIVE reviews: {metrics['actual_positive']}")
    print(f"Actual NEGATIVE reviews: {metrics['actual_negative']}")
    print(f"Recall for actual POSITIVE reviews: {metrics['positive_recall']:.2%}")
    print(f"Recall for actual NEGATIVE reviews: {metrics['negative_recall']:.2%}")
    print(f"True positives: {metrics['true_positives']}")
    print(f"True negatives: {metrics['true_negatives']}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.2%}")

    errors = [row for row in rows if row['correct'] != 'True']
    print('\nIncorrect predictions')
    if not errors:
        print('None')
        return metrics
    headers = ('row_id', 'rating', 'title', 'abbreviated text', 'actual', 'predicted')
    table = []
    for row in errors:
        table.append((
            row['row_id'],
            row['rating'],
            textwrap.shorten(row['title'], width=30, placeholder='...'),
            textwrap.shorten(row['text'], width=60, placeholder='...'),
            row['actual_sentiment'],
            row['predicted_sentiment'],
        ))
    widths = [max(len(str(value)) for value in [header] + [r[i] for r in table])
              for i, header in enumerate(headers)]
    print(' | '.join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print('-+-'.join('-' * width for width in widths))
    for values in table:
        print(' | '.join(str(value).ljust(widths[i]) for i, value in enumerate(values)))
    return metrics


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Evaluate the first 100 reviews')
    parser.add_argument('--data', type=Path,
                        default=project_root / 'data/Gift_Cards.jsonl')
    parser.add_argument('--output', type=Path,
                        default=project_root / 'outputs/predictions.csv')
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    try:
        rows = evaluate(args.data, args.output, args.limit)
    except Exception as exc:
        print(f'\nEvaluation stopped: {exc}', file=sys.stderr)
        print('Completed predictions remain saved; rerun the same command to resume.',
              file=sys.stderr)
        raise SystemExit(1) from exc
    print_report(rows)


if __name__ == '__main__':
    main()
