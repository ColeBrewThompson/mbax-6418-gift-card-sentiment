# MBAX 6418: Gift Card sentiment

## Current evaluation: Steps 6–7

The current dashboard uses a fixed-seed, balanced **three-class** sample from
the entire JSONL file. It selects 50 reviews with 4–5 stars (POSITIVE), 50 with
3 stars (NEUTRAL), and 50 with 1–2 stars (NEGATIVE). The seed is `6418` and the
CSV retains each review's original source row number. Selection uses ratings
locally; the model receives only title and text. Its response includes both
three-class sentiment and primary emotion.

```bash
python3 -m giftcard_reviews.balanced_evaluation
python3 scripts/build_balanced_dashboard.py
python3 scripts/verify_dashboard.py
node scripts/verify_dashboard_filters.mjs
python3 -m unittest discover -s tests -v
```

The resumable model output is `outputs/balanced_three_class.csv`. Re-running the
evaluation with the same seed validates and reuses completed rows. The offline
dashboard is `dashboard/index.html`; open it directly in a browser. Balanced
accuracy is the mean of POSITIVE, NEUTRAL, and NEGATIVE recall. The three-class
confusion matrix and the 3-star NEUTRAL row show whether mixed reviews get their
own class or drift into POSITIVE or NEGATIVE.

The Step 7 visual summary draws three charts from that CSV: star-rating counts,
actual versus predicted class totals, and correct versus incorrect predictions
within each actual class. Counts are printed beside the bars. A browser-side
check compares every chart count with the embedded reviews and confirms that
each nonzero segment occupies visible pixels. `scripts/verify_dashboard.py`
independently compares the embedded reviews and chart values to the saved CSV;
`scripts/verify_dashboard_filters.mjs` checks the live explorer filters.

On this seeded sample, the model matched **114/150 labels (76.00%)**. Recall was
**96.00% POSITIVE**, **36.00% NEUTRAL**, and **96.00% NEGATIVE**. Of the 50 actual
NEUTRAL reviews, it predicted 18 NEUTRAL, 9 POSITIVE, and 23 NEGATIVE. Balanced
accuracy is **76.00%**. The 3-star reviews now have their own label, but most of
them were predicted as a neighboring polarity from the review language.

The old first-100 binary outputs, `outputs/predictions.csv` and
`outputs/predictions_with_emotions.csv`, are retained for comparison. Their
98% binary accuracy was measured on an overwhelmingly positive, differently
labeled sample and is not directly comparable to the balanced three-class run.
The previous dashboard HTML is preserved as `dashboard/step3_binary.html`.

## Earlier milestones

Small Python 3.9+ project using only the standard library. No installation required.

From this project folder:

```bash
python3 -m giftcard_reviews --inspect-only
python3 -m giftcard_reviews
python3 -m giftcard_reviews.evaluation  # historical binary first-100 run
python3 scripts/build_dashboard.py  # historical Step 3 builder; replaces current dashboard
python3 -m unittest discover -s tests -v
```

The default run validates every JSON Lines record, prints three text samples,
and sends the first 10 reviews to the course endpoint, one at a time. It prints
each title, shortened text, three-class sentiment, and primary emotion. Full
title and text are sent to the model; shortening is only for display. Choose a different input with
`--data PATH` or batch size with `--limit N`. Selection follows file order and
does not depend on rating. This is a loading/inference smoke test, not an accuracy
estimate or representative evaluation.

## Modules

- `giftcard_reviews/dataset.py`: streams records, checks required fields and types,
  and reports malformed records with line numbers. Other fields are allowed.
- `giftcard_reviews/sentiment.py`: historical binary classifiers and the current
  `ThreeClassEmotionClassifier`, each accepting title and text only.
- `giftcard_reviews/__main__.py`: validates the dataset and prints samples/results.
- `giftcard_reviews/evaluation.py`: resumably evaluates the first 100 rows, writes
  `outputs/predictions.csv`, and prints metrics plus every classification error.
- `giftcard_reviews/emotion_evaluation.py`: resumably creates the separate Step 5
  sentiment/emotion output and compares the LLM and NRC emotion methods.
- `giftcard_reviews/balanced_evaluation.py`: fixed-seed balanced selection,
  resumable three-class scoring, and macro-recall/confusion-matrix reporting.
- `giftcard_reviews/nrc_emotion.py`: independent NRC word-list tokenizer, scorer,
  deterministic tie handling, and explicit no-match handling.
- `tests/test_core.py`: offline request-boundary, parser, fallback, and loader checks.

Example reuse:

```python
from giftcard_reviews.sentiment import ThreeClassEmotionClassifier
result = ThreeClassEmotionClassifier().classify(
    title='Great gift', text='Easy to use and appreciated.'
)
```

Only title and text are passed as review data to the classifier and endpoint.
The classifier has no rating argument and receives no review dictionary.
Metadata stays in the local loader. The prompt treats review instructions as
untrusted content and asks the model to ignore star ratings mentioned in prose.

Endpoint: `http://dobolyi.com:9001/v1/chat/completions`

Model: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`

The supplied course API key defaults to `6418`; optionally override it with the
`GIFT_CARD_API_KEY` environment variable.

The client first requests JSON Schema output. If the endpoint explicitly rejects
the response format with HTTP 400/422, it tries JSON object mode, then prompt-only
JSON. The current response must contain exactly `sentiment` (POSITIVE, NEUTRAL,
or NEGATIVE) and `primary_emotion` (one of the eight NRC emotion names). Invalid
or truncated responses raise an error; network/authentication failures do not
become predictions. The historical binary classifiers keep their original
two-label prompts and response validation.

Response format reference: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Compatibility with the third-party server is checked through a live run.

Initial inspection: 152,410 records; all five required fields present in all
records. Observed types: title/text strings, rating float, verified_purchase
Boolean, helpful_vote integer.

The Step 2 command writes and flushes each prediction before requesting the next
one. If an API call still fails after three attempts, rerun the command to resume
from the saved rows. Existing output is checked against the current source data
before it is reused. The classifier receives only title and text; the rating-based
ground truth is created locally after each prediction returns.

## Historical Step 3 dashboard

The historical builder, `scripts/build_dashboard.py`, can regenerate the binary
dashboard from `outputs/predictions.csv`. It overwrites `dashboard/index.html`;
rerun `scripts/build_balanced_dashboard.py` to restore the current Step 6 view.
Both dashboard versions are self-contained and require no server or network.

The dashboard includes verified headline metrics, class balance, class recall,
the confusion matrix, performance by star rating, detailed error analysis, and
a searchable/filterable review explorer. To regenerate it from the CSV source of
truth, run `python3 scripts/build_dashboard.py`.

## Step 5 emotion evaluation

Download the NRC Emotion Lexicon for educational use from the
[official NRC Emotion Lexicon page](https://www.saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm),
then extract `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`. The lexicon is not bundled
because its terms prohibit redistribution. This project uses the NRC Emotion
Lexicon created by Saif M. Mohammad and Peter Turney at the National Research
Council Canada.

Run the separate, resumable emotion evaluation with:

```bash
python3 -m giftcard_reviews.emotion_evaluation \
  --lexicon /path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
python3 scripts/verify_emotion_evaluation.py \
  --lexicon /path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
```

The command writes `outputs/predictions_with_emotions.csv`. It preserves the
original Step 2 output, reports any changed sentiment predictions, counts NRC
ties and no-match cases explicitly, and prints cross-method agreement statistics.
For a tied highest nonzero NRC score, the deterministic priority is anger,
anticipation, disgust, fear, joy, sadness, surprise, then trust. All tied winners
are retained in `nrc_tied_emotions`; zero-score reviews use `NO_MATCH`.
