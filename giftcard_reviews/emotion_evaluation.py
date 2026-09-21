"""Resumable Step 5 evaluation of LLM and NRC primary emotions."""
import argparse
import csv
import os
import sys
import textwrap
from collections import Counter
from pathlib import Path

from .evaluation import CSV_FIELDS, actual_sentiment, classify_with_retries, first_reviews
from .nrc_emotion import NRC_EMOTIONS, load_nrc_lexicon, score_emotions
from .sentiment import EMOTIONS, LABELS, SentimentEmotionClassifier


NRC_SCORE_FIELDS = tuple(f'nrc_{emotion}' for emotion in NRC_EMOTIONS)
EMOTION_CSV_FIELDS = (
    *CSV_FIELDS,
    'llm_emotion',
    *NRC_SCORE_FIELDS,
    'nrc_emotion',
    'nrc_tie',
    'nrc_tied_emotions',
    'nrc_no_match',
)


def build_row(review, row_id, model_result, nrc_result):
    """Build local fields only after the title/text-only model call returns."""
    predicted = model_result['sentiment']
    actual = actual_sentiment(review['rating'])
    row = {
        'row_id': str(row_id),
        'title': review['title'],
        'text': review['text'],
        'rating': str(review['rating']),
        'actual_sentiment': actual,
        'predicted_sentiment': predicted,
        'correct': str(predicted == actual),
        'verified_purchase': str(review['verified_purchase']),
        'helpful_vote': str(review['helpful_vote']),
        'llm_emotion': model_result['primary_emotion'],
        'nrc_emotion': nrc_result['emotion'],
        'nrc_tie': str(nrc_result['tie']),
        'nrc_tied_emotions': '|'.join(nrc_result['tied_emotions']),
        'nrc_no_match': str(nrc_result['no_match']),
    }
    for emotion in NRC_EMOTIONS:
        row[f'nrc_{emotion}'] = str(nrc_result['scores'][emotion])
    return {field: row[field] for field in EMOTION_CSV_FIELDS}


def load_completed(output_path, reviews, lexicon):
    output_path = Path(output_path)
    if not output_path.exists():
        return {}
    with output_path.open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != EMOTION_CSV_FIELDS:
            raise ValueError(f'{output_path} has an unexpected CSV header')
        completed = {}
        for csv_line, row in enumerate(reader, 2):
            try:
                row_id = int(row['row_id'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{output_path}, line {csv_line}: invalid row_id') from exc
            if row_id < 1 or row_id > len(reviews) or row_id in completed:
                raise ValueError(f'{output_path}, line {csv_line}: unexpected row_id {row_id}')
            model_result = {
                'sentiment': row['predicted_sentiment'],
                'primary_emotion': row['llm_emotion'],
            }
            if model_result['sentiment'] not in LABELS or model_result['primary_emotion'] not in EMOTIONS:
                raise ValueError(f'{output_path}, line {csv_line}: invalid model label')
            review = reviews[row_id - 1]
            nrc_result = score_emotions(review['title'], review['text'], lexicon)
            expected = build_row(review, row_id, model_result, nrc_result)
            if row != expected:
                raise ValueError(f'{output_path}, line {csv_line}: row does not match source data')
            completed[row_id] = row
    return completed


def append_row(output_path, row):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists()
    with output_path.open('a', newline='', encoding='utf-8') as destination:
        writer = csv.DictWriter(destination, fieldnames=EMOTION_CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        destination.flush()
        os.fsync(destination.fileno())


def evaluate_emotions(data_path, output_path, lexicon_path, limit=100, classifier=None):
    reviews = first_reviews(data_path, limit)
    lexicon = load_nrc_lexicon(lexicon_path)
    completed = load_completed(output_path, reviews, lexicon)
    classifier = classifier or SentimentEmotionClassifier()
    if completed:
        print(f'Resuming with {len(completed)} of {limit} rows already saved.', flush=True)

    for row_id, review in enumerate(reviews, 1):
        if row_id in completed:
            continue
        # The model boundary receives only these two strings. Rating, prior
        # predictions, ground truth, and NRC scores are not arguments or payload.
        model_result = classify_with_retries(
            classifier,
            title=review['title'],
            text=review['text'],
        )
        nrc_result = score_emotions(review['title'], review['text'], lexicon)
        row = build_row(review, row_id, model_result, nrc_result)
        append_row(output_path, row)
        completed[row_id] = row
        print(f'Completed {len(completed)}/{limit}: row {row_id}', flush=True)

    return [completed[row_id] for row_id in range(1, limit + 1)]


def compare_sentiments(rows, original_path):
    with Path(original_path).open(newline='', encoding='utf-8') as source:
        original = {int(row['row_id']): row for row in csv.DictReader(source)}
    if set(original) != {int(row['row_id']) for row in rows}:
        raise ValueError('Original predictions do not cover the same row IDs')
    changed = []
    for row in rows:
        row_id = int(row['row_id'])
        old = original[row_id]
        if old['title'] != row['title'] or old['text'] != row['text']:
            raise ValueError(f'Original prediction row {row_id} has different review text')
        if old['predicted_sentiment'] != row['predicted_sentiment']:
            changed.append({
                'row_id': row_id,
                'title': row['title'],
                'original': old['predicted_sentiment'],
                'new': row['predicted_sentiment'],
            })
    return changed


def calculate_comparison(rows):
    comparable = [row for row in rows if row['nrc_no_match'] != 'True']
    agreements = [row for row in comparable if row['llm_emotion'] == row['nrc_emotion']]
    disagreements = [row for row in comparable if row['llm_emotion'] != row['nrc_emotion']]
    llm_counts = Counter(row['llm_emotion'] for row in rows)
    nrc_counts = Counter(row['nrc_emotion'] for row in rows)
    pairs = Counter((row['llm_emotion'], row['nrc_emotion']) for row in disagreements)

    by_llm = {}
    by_nrc = {}
    for emotion in EMOTIONS:
        llm_group = [row for row in comparable if row['llm_emotion'] == emotion]
        nrc_group = [row for row in comparable if row['nrc_emotion'] == emotion]
        by_llm[emotion] = (
            sum(row['nrc_emotion'] == emotion for row in llm_group), len(llm_group)
        )
        by_nrc[emotion] = (
            sum(row['llm_emotion'] == emotion for row in nrc_group), len(nrc_group)
        )
    return {
        'comparable': len(comparable),
        'agreeing': len(agreements),
        'disagreeing': len(disagreements),
        'agreement_percentage': len(agreements) / len(comparable) if comparable else 0.0,
        'no_match': len(rows) - len(comparable),
        'ties': sum(row['nrc_tie'] == 'True' for row in rows),
        'llm_counts': llm_counts,
        'nrc_counts': nrc_counts,
        'by_llm': by_llm,
        'by_nrc': by_nrc,
        'pairs': pairs,
        'agreements': agreements,
        'disagreements': disagreements,
    }


def _ratio(agree, total):
    return f'{agree}/{total} ({agree / total:.2%})' if total else '0/0 (n.a.)'


def score_summary(row):
    return ' '.join(f'{emotion[:3]}={row[f"nrc_{emotion}"]}' for emotion in NRC_EMOTIONS)


def print_comparison(rows, original_path):
    changed = compare_sentiments(rows, original_path)
    result = calculate_comparison(rows)
    print('\nSentiment stability')
    print(f'Changed sentiment predictions: {len(changed)}')
    for item in changed:
        print(f"  Row {item['row_id']}: {item['original']} -> {item['new']} | {item['title']}")

    print('\nEmotion comparison')
    print(f"Comparable reviews: {result['comparable']}")
    print(f"Agreeing: {result['agreeing']}")
    print(f"Disagreeing: {result['disagreeing']}")
    print(f"Agreement: {result['agreement_percentage']:.2%}")
    print(f"NRC no-match reviews (excluded from agreement): {result['no_match']}")
    print(f"NRC ties: {result['ties']}")

    print('\nCounts by emotion')
    print('Emotion       LLM  NRC')
    print('------------  ---  ---')
    for emotion in EMOTIONS:
        print(f'{emotion:<12}  {result["llm_counts"][emotion]:>3}  {result["nrc_counts"][emotion]:>3}')
    print(f'{"NO_MATCH":<12}  {0:>3}  {result["nrc_counts"]["NO_MATCH"]:>3}')

    print('\nAgreement by LLM emotion')
    for emotion in EMOTIONS:
        print(f'  {emotion:<12} {_ratio(*result["by_llm"][emotion])}')
    print('\nAgreement by NRC emotion')
    for emotion in EMOTIONS:
        print(f'  {emotion:<12} {_ratio(*result["by_nrc"][emotion])}')

    print('\nMost common comparable disagreement pairs (LLM -> NRC)')
    for (llm, nrc), count in result['pairs'].most_common():
        print(f'  {llm:<12} -> {nrc:<12} {count}')

    flagged = [
        row for row in rows
        if (row['nrc_no_match'] == 'True' or row['nrc_tie'] == 'True'
            or row['llm_emotion'] != row['nrc_emotion'])
    ]
    print('\nDisagreements, ties, and NRC no-match cases')
    headers = ('row', 'rating', 'title', 'abbreviated text', 'sentiment',
               'LLM', 'NRC', 'NRC scores', 'status')
    table = []
    for row in flagged:
        status = ('NO_MATCH' if row['nrc_no_match'] == 'True'
                  else f'TIE:{row["nrc_tied_emotions"]}' if row['nrc_tie'] == 'True'
                  else 'DISAGREE')
        table.append((
            row['row_id'], row['rating'],
            textwrap.shorten(row['title'], width=25, placeholder='...'),
            textwrap.shorten(row['text'], width=48, placeholder='...'),
            row['predicted_sentiment'], row['llm_emotion'], row['nrc_emotion'],
            score_summary(row), status,
        ))
    widths = [max(len(str(value)) for value in [headers[i]] + [r[i] for r in table])
              for i in range(len(headers))]
    print(' | '.join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print('-+-'.join('-' * width for width in widths))
    for values in table:
        print(' | '.join(str(values[i]).ljust(widths[i]) for i in range(len(headers))))
    return result, changed


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Evaluate LLM and NRC emotions')
    parser.add_argument('--data', type=Path,
                        default=project_root / 'data/Gift_Cards.jsonl')
    parser.add_argument('--output', type=Path,
                        default=project_root / 'outputs/predictions_with_emotions.csv')
    parser.add_argument('--original', type=Path,
                        default=project_root / 'outputs/predictions.csv')
    parser.add_argument('--lexicon', type=Path, required=True,
                        help='Official NRC word-level v0.92 text file')
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    try:
        rows = evaluate_emotions(args.data, args.output, args.lexicon, args.limit)
        print_comparison(rows, args.original)
    except Exception as exc:
        print(f'\nEmotion evaluation stopped: {exc}', file=sys.stderr)
        print('Completed rows remain saved; rerun the command to resume.', file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
