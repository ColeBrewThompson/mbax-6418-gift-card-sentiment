"""Reproducible, resumable three-class evaluation over the full review file."""
import argparse
import csv
import os
import random
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .dataset import read_reviews
from .evaluation import classify_with_retries
from .sentiment import EMOTIONS, THREE_CLASS_LABELS, ThreeClassEmotionClassifier

SEED = 6418
PER_CLASS = 50
FIELDS = (
    'sample_id', 'row_id', 'title', 'text', 'rating', 'actual_sentiment',
    'predicted_sentiment', 'correct', 'verified_purchase', 'helpful_vote',
    'llm_emotion',
)


def actual_sentiment(rating):
    if rating not in (1, 2, 3, 4, 5):
        raise ValueError(f'Invalid star rating: {rating}')
    if rating >= 4:
        return 'POSITIVE'
    if rating == 3:
        return 'NEUTRAL'
    return 'NEGATIVE'


def select_balanced_reviews(path, per_class=PER_CLASS, seed=SEED):
    """Sample source line IDs uniformly within each class, then read them in order."""
    if per_class < 1:
        raise ValueError('per_class must be positive')
    by_class = {label: [] for label in THREE_CLASS_LABELS}
    for row_id, review in enumerate(read_reviews(path), 1):
        by_class[actual_sentiment(review['rating'])].append(row_id)
    rng = random.Random(seed)
    chosen = set()
    for label in THREE_CLASS_LABELS:
        if len(by_class[label]) < per_class:
            raise ValueError(f'Only {len(by_class[label])} {label} reviews available')
        chosen.update(rng.sample(by_class[label], per_class))
    selected = []
    for row_id, review in enumerate(read_reviews(path), 1):
        if row_id in chosen:
            selected.append((row_id, review))
    assert len(selected) == per_class * len(THREE_CLASS_LABELS)
    return selected


def expected_row(sample_id, row_id, review, prediction):
    actual = actual_sentiment(review['rating'])
    row = {
        'sample_id': str(sample_id),
        'row_id': str(row_id),
        'title': review['title'],
        'text': review['text'],
        'rating': str(review['rating']),
        'actual_sentiment': actual,
        'predicted_sentiment': prediction['sentiment'],
        'correct': str(actual == prediction['sentiment']),
        'verified_purchase': str(review['verified_purchase']),
        'helpful_vote': str(review['helpful_vote']),
        'llm_emotion': prediction['primary_emotion'],
    }
    return {field: row[field] for field in FIELDS}


def load_completed(output_path, selected):
    path = Path(output_path)
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f'{path} has an unexpected header')
        completed = {}
        for csv_line, row in enumerate(reader, 2):
            try:
                sample_id = int(row['sample_id'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path}, line {csv_line}: invalid sample_id') from exc
            if sample_id < 1 or sample_id > len(selected) or sample_id in completed:
                raise ValueError(f'{path}, line {csv_line}: duplicate/unexpected sample_id')
            if (row['predicted_sentiment'] not in THREE_CLASS_LABELS
                    or row['llm_emotion'] not in EMOTIONS):
                raise ValueError(f'{path}, line {csv_line}: invalid model label')
            row_id, review = selected[sample_id - 1]
            prediction = {'sentiment': row['predicted_sentiment'],
                          'primary_emotion': row['llm_emotion']}
            if row != expected_row(sample_id, row_id, review, prediction):
                raise ValueError(f'{path}, line {csv_line}: saved row differs from source')
            completed[sample_id] = row
    return completed


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        if header_needed:
            writer.writeheader()
        writer.writerow(row)
        target.flush()
        os.fsync(target.fileno())


def order_completed_output(path, completed):
    """Present a finished concurrent run in source/sample order, atomically."""
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8',
                                     dir=path.parent, prefix='.balanced-',
                                     suffix='.csv', delete=False) as target:
        temporary = Path(target.name)
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for sample_id in sorted(completed):
            writer.writerow(completed[sample_id])
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def evaluate(data_path, output_path, per_class=PER_CLASS, seed=SEED, classifier=None,
             workers=4):
    selected = select_balanced_reviews(data_path, per_class, seed)
    completed = load_completed(output_path, selected)
    if workers < 1:
        raise ValueError('workers must be positive')
    print(f'Selected {len(selected)} reviews with seed {seed}: '
          f'{per_class} per class. Resuming {len(completed)} saved results.', flush=True)
    def predict(review):
        # A separate client per worker avoids shared format-negotiation state.
        client = classifier if classifier is not None else ThreeClassEmotionClassifier()
        # No rating, label, old prediction, or NRC score crosses this boundary.
        return classify_with_retries(client, title=review['title'], text=review['text'])

    pending = [(sample_id, row_id, review)
               for sample_id, (row_id, review) in enumerate(selected, 1)
               if sample_id not in completed]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(predict, review): (sample_id, row_id, review)
                   for sample_id, row_id, review in pending}
        for future in as_completed(futures):
            sample_id, row_id, review = futures[future]
            try:
                prediction = future.result()
                row = expected_row(sample_id, row_id, review, prediction)
                append_row(output_path, row)
                completed[sample_id] = row
                print(f'Completed {len(completed)}/{len(selected)}: source row {row_id}',
                      flush=True)
            except Exception as exc:
                failures.append((row_id, exc))
                print(f'Failed source row {row_id}: {exc}', file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} model calls failed; rerun to resume')
    order_completed_output(output_path, completed)
    return [completed[index] for index in range(1, len(selected) + 1)]


def metrics(rows):
    matrix = Counter((r['actual_sentiment'], r['predicted_sentiment']) for r in rows)
    actual_counts = Counter(r['actual_sentiment'] for r in rows)
    predicted_counts = Counter(r['predicted_sentiment'] for r in rows)
    recalls = {
        label: matrix[label, label] / actual_counts[label] if actual_counts[label] else 0
        for label in THREE_CLASS_LABELS
    }
    correct = sum(matrix[label, label] for label in THREE_CLASS_LABELS)
    return {
        'total': len(rows), 'correct': correct, 'incorrect': len(rows) - correct,
        'accuracy': correct / len(rows) if rows else 0,
        'balanced_accuracy': sum(recalls.values()) / 3,
        'matrix': matrix, 'actual_counts': actual_counts,
        'predicted_counts': predicted_counts, 'recalls': recalls,
    }


def print_report(rows):
    result = metrics(rows)
    print(f"\nThree-class accuracy: {result['correct']}/{result['total']} "
          f"({result['accuracy']:.2%})")
    print(f"Balanced accuracy (macro recall): {result['balanced_accuracy']:.2%}")
    print('Actual → predicted | POSITIVE  NEUTRAL  NEGATIVE | Recall')
    for actual in THREE_CLASS_LABELS:
        cells = '  '.join(f'{result["matrix"][actual, pred]:>8}'
                          for pred in THREE_CLASS_LABELS)
        print(f'{actual:<18} {cells} | {result["recalls"][actual]:.2%}')
    print('Predicted totals: ' + ', '.join(
        f'{label} {result["predicted_counts"][label]}' for label in THREE_CLASS_LABELS
    ))
    print('Errors by rating: ' + ', '.join(
        f'{rating}★ {sum(float(r["rating"]) == rating and r["correct"] == "False" for r in rows)}'
        for rating in range(1, 6)
    ))
    return result


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Balanced three-class review evaluation')
    parser.add_argument('--data', type=Path, default=root / 'data/Gift_Cards.jsonl')
    parser.add_argument('--output', type=Path,
                        default=root / 'outputs/balanced_three_class.csv')
    parser.add_argument('--per-class', type=int, default=PER_CLASS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    try:
        rows = evaluate(args.data, args.output, args.per_class, args.seed,
                        workers=args.workers)
    except Exception as exc:
        print(f'Evaluation stopped: {exc}', file=sys.stderr)
        print('Saved predictions are intact; rerun the same command to resume.',
              file=sys.stderr)
        raise SystemExit(1) from exc
    print_report(rows)


if __name__ == '__main__':
    main()
