"""Independently verify the Step 5 CSV against source data and the NRC lexicon."""
import argparse
import csv
import sys
from itertools import islice
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from giftcard_reviews.dataset import read_reviews
from giftcard_reviews.emotion_evaluation import (
    EMOTION_CSV_FIELDS,
    build_row,
    calculate_comparison,
    compare_sentiments,
)
from giftcard_reviews.nrc_emotion import load_nrc_lexicon, score_emotions
from giftcard_reviews.sentiment import EMOTIONS, LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexicon', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/Gift_Cards.jsonl')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/predictions_with_emotions.csv')
    parser.add_argument('--original', type=Path,
                        default=ROOT / 'outputs/predictions.csv')
    args = parser.parse_args()

    reviews = list(islice(read_reviews(args.data), 100))
    lexicon = load_nrc_lexicon(args.lexicon)
    with args.output.open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames or ()) == EMOTION_CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 100
    assert [int(row['row_id']) for row in rows] == list(range(1, 101))

    for row_id, (review, row) in enumerate(zip(reviews, rows), 1):
        assert row['predicted_sentiment'] in LABELS
        assert row['llm_emotion'] in EMOTIONS
        nrc_result = score_emotions(review['title'], review['text'], lexicon)
        expected = build_row(
            review,
            row_id,
            {'sentiment': row['predicted_sentiment'],
             'primary_emotion': row['llm_emotion']},
            nrc_result,
        )
        assert row == expected, f'Row {row_id} differs from independently rebuilt data'

    changed = compare_sentiments(rows, args.original)
    assert changed == [], f'Sentiment predictions changed: {changed}'
    comparison = calculate_comparison(rows)
    assert comparison['comparable'] == 85
    assert comparison['agreeing'] == 20
    assert comparison['disagreeing'] == 65
    assert f"{comparison['agreement_percentage']:.2%}" == '23.53%'
    assert comparison['no_match'] == 15
    assert comparison['ties'] == 52

    print('PASS: 100 sequential Step 5 rows and all required columns verified')
    print('PASS: title, text, local metadata, sentiment labels, and correctness verified')
    print('PASS: every NRC score, winner, tie, tied-emotion list, and no-match flag rebuilt')
    print('PASS: original and new sentiment predictions match on all 100 rows')
    print('PASS: comparable/agreement/disagreement/no-match/tie totals verified')


if __name__ == '__main__':
    main()
