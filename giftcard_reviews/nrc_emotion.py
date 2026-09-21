"""Independent word-list emotion scoring with the NRC Emotion Lexicon."""
import html
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


NRC_EMOTIONS = (
    'anger', 'anticipation', 'disgust', 'fear',
    'joy', 'sadness', 'surprise', 'trust',
)
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def load_nrc_lexicon(path):
    """Load only positive associations for the eight requested emotions."""
    associations = defaultdict(set)
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) != 3:
                raise ValueError(f'{path}, line {line_number}: expected three fields')
            word, emotion, associated = parts
            if emotion in NRC_EMOTIONS and associated == '1':
                associations[word.casefold()].add(emotion)
            elif associated not in ('0', '1'):
                raise ValueError(f'{path}, line {line_number}: invalid association value')
    if not associations:
        raise ValueError(f'{path}: no NRC emotion associations found')
    return dict(associations)


def tokenize(title, text):
    """Normalize Unicode/HTML entities and extract lowercase English word tokens."""
    combined = unicodedata.normalize('NFKC', f'{title} {text}')
    normalized = html.unescape(combined).casefold()
    return TOKEN_PATTERN.findall(normalized)


def score_emotions(title, text, lexicon):
    """Count matched tokens and select a transparent deterministic winner."""
    scores = {emotion: 0 for emotion in NRC_EMOTIONS}
    for token in tokenize(title, text):
        for emotion in lexicon.get(token, ()):
            scores[emotion] += 1

    maximum = max(scores.values())
    if maximum == 0:
        return {
            'emotion': 'NO_MATCH',
            'scores': scores,
            'tie': False,
            'tied_emotions': (),
            'no_match': True,
        }

    winners = tuple(emotion for emotion in NRC_EMOTIONS if scores[emotion] == maximum)
    return {
        # Fixed priority follows NRC_EMOTIONS and is recorded when a tie occurs.
        'emotion': winners[0].upper(),
        'scores': scores,
        'tie': len(winners) > 1,
        'tied_emotions': tuple(emotion.upper() for emotion in winners),
        'no_match': False,
    }
