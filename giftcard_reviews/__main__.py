"""Run with: python3 -m giftcard_reviews"""
import argparse
from pathlib import Path
import textwrap

from .dataset import REQUIRED_TYPES, read_reviews
from .sentiment import ThreeClassEmotionClassifier


def main():
    parser = argparse.ArgumentParser(description='Inspect and classify Gift Card reviews')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent.parent / 'data/Gift_Cards.jsonl')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    selected = []
    count = 0
    for review in read_reviews(args.data):
        count += 1
        if len(selected) < max(3, args.limit):
            selected.append(review)
    if not count:
        parser.error('dataset is empty')
    print(f'Validated {count:,} reviews. Required fields: {", ".join(REQUIRED_TYPES)}', flush=True)
    print('\nSample reviews (title and text):', flush=True)
    for review in selected[:3]:
        print(f'\nTitle: {review["title"]}\nText: {textwrap.shorten(review["text"], width=240)}', flush=True)
    if args.inspect_only:
        return
    classifier = ThreeClassEmotionClassifier()
    print(f'\nClassifying {min(args.limit, count)} reviews:', flush=True)
    for index, review in enumerate(selected[:args.limit], 1):
        prediction = classifier.classify(title=review['title'], text=review['text'])
        print(f'\n{index}. Title: {review["title"]}\n'
              f'Text: {textwrap.shorten(review["text"], width=180)}\n'
              f'Predicted sentiment: {prediction["sentiment"]}\n'
              f'Primary emotion: {prediction["primary_emotion"]}', flush=True)
    print(f'\nResponse format used: {classifier.response_mode}', flush=True)


if __name__ == '__main__':
    main()
