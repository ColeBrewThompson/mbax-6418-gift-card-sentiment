"""Build the offline Step 3 dashboard from outputs/predictions.csv."""
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / 'outputs/predictions.csv'
OUTPUT_PATH = PROJECT_ROOT / 'dashboard/index.html'


def load_rows():
    with INPUT_PATH.open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    required = {
        'row_id', 'title', 'text', 'rating', 'actual_sentiment',
        'predicted_sentiment', 'correct', 'verified_purchase', 'helpful_vote',
    }
    if len(rows) != 100 or not rows or not required.issubset(rows[0]):
        raise ValueError('Expected 100 prediction rows with all required columns')
    for row in rows:
        row['row_id'] = int(row['row_id'])
        row['rating'] = int(float(row['rating']))
        row['correct'] = row['correct'].lower() == 'true'
        row['verified_purchase'] = row['verified_purchase'].lower() == 'true'
        row['helpful_vote'] = int(row['helpful_vote'])
    if [row['row_id'] for row in rows] != list(range(1, 101)):
        raise ValueError('Expected sequential row IDs from 1 through 100')
    return rows


def calculate(rows):
    confusion = Counter((r['actual_sentiment'], r['predicted_sentiment']) for r in rows)
    tp = confusion['POSITIVE', 'POSITIVE']
    tn = confusion['NEGATIVE', 'NEGATIVE']
    fp = confusion['NEGATIVE', 'POSITIVE']
    fn = confusion['POSITIVE', 'NEGATIVE']
    actual_positive = tp + fn
    actual_negative = tn + fp
    positive_recall = tp / actual_positive
    negative_recall = tn / actual_negative
    by_rating = defaultdict(lambda: {'count': 0, 'correct': 0, 'errors': 0})
    for row in rows:
        bucket = by_rating[row['rating']]
        bucket['count'] += 1
        bucket['correct'] += row['correct']
        bucket['errors'] += not row['correct']
    for rating in range(1, 6):
        bucket = by_rating[rating]
        bucket['accuracy'] = bucket['correct'] / bucket['count'] if bucket['count'] else None
    return {
        'total': len(rows),
        'correct': tp + tn,
        'incorrect': fp + fn,
        'accuracy': (tp + tn) / len(rows),
        'actual_positive': actual_positive,
        'actual_negative': actual_negative,
        'positive_recall': positive_recall,
        'negative_recall': negative_recall,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'balanced_accuracy': (positive_recall + negative_recall) / 2,
        'by_rating': by_rating,
    }


def error_cards(rows):
    explanations = {
        18: (
            'Nuanced disagreement',
            'The review opens with praise and carries a 5-star rating, but most of '
            'the text describes a recurring missing-note problem. The model appears '
            'to have weighted the substantial complaint more heavily than the '
            'reviewer’s overall positive judgment.'
        ),
        99: (
            'Label-rule mismatch',
            'The wording is plainly positive: the reviewer calls the product very '
            'easy to use. The assignment rule nevertheless maps its 3-star rating '
            'to NEGATIVE, so the model’s POSITIVE prediction conflicts with the '
            'rating-derived label rather than with the language itself.'
        ),
    }
    cards = []
    for row in rows:
        if row['correct']:
            continue
        label, explanation = explanations[row['row_id']]
        cards.append(f'''<article class="error-card">
          <div class="error-card__top">
            <div><span class="row-number">Row {row['row_id']}</span>
              <h3>{html.escape(row['title'])}</h3></div>
            <span class="error-type">{label}</span>
          </div>
          <blockquote>{html.escape(row['text'])}</blockquote>
          <div class="label-comparison" aria-label="Sentiment comparison">
            <div><span>Rating</span><strong>{row['rating']} star{'s' if row['rating'] != 1 else ''}</strong></div>
            <div><span>Rating-derived actual</span><strong class="pill pill--{'positive' if row['actual_sentiment'] == 'POSITIVE' else 'negative'}">{row['actual_sentiment']}</strong></div>
            <div><span>Model prediction</span><strong class="pill pill--{'positive' if row['predicted_sentiment'] == 'POSITIVE' else 'negative'}">{row['predicted_sentiment']}</strong></div>
          </div>
          <p class="error-explanation">{explanation}</p>
        </article>''')
    return '\n'.join(cards)


def rating_rows(stats):
    output = []
    for rating in range(1, 6):
        bucket = stats['by_rating'][rating]
        accuracy = bucket['accuracy'] * 100 if bucket['accuracy'] is not None else 0
        width = bucket['count'] / stats['total'] * 100 if stats['total'] else 0
        highlight = ' rating-row--highlight' if rating == 3 else ''
        output.append(f'''<tr class="rating-row{highlight}">
          <th scope="row"><span class="stars" aria-label="{rating} stars">{'★' * rating}<span>{'★' * (5-rating)}</span></span></th>
          <td><div class="count-cell"><strong>{bucket['count']}</strong><span class="mini-bar"><i style="width:{width:.2f}%"></i></span></div></td>
          <td><div class="accuracy-cell"><strong>{accuracy:.2f}%</strong><span class="mini-bar mini-bar--accuracy"><i style="width:{accuracy:.2f}%"></i></span></div></td>
          <td><span class="error-count{' has-error' if bucket['errors'] else ''}">{bucket['errors']}</span></td>
        </tr>''')
    return '\n'.join(output)


def build():
    rows = load_rows()
    stats = calculate(rows)
    expected = {
        'total': 100, 'correct': 98, 'incorrect': 2,
        'actual_positive': 93, 'actual_negative': 7,
        'tp': 92, 'tn': 6, 'fp': 1, 'fn': 1,
    }
    for key, value in expected.items():
        if stats[key] != value:
            raise ValueError(f'Unexpected {key}: {stats[key]} (expected {value})')
    if round(stats['accuracy'] * 100, 2) != 98.00:
        raise ValueError('Unexpected overall accuracy')
    if round(stats['balanced_accuracy'] * 100, 2) != 92.32:
        raise ValueError('Unexpected balanced accuracy')
    errors = [row for row in rows if not row['correct']]
    if [row['row_id'] for row in errors] != [18, 99]:
        raise ValueError('Expected errors on rows 18 and 99')

    data_json = json.dumps(rows, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    document = TEMPLATE
    replacements = {
        '__DATA_JSON__': data_json,
        '__ERROR_CARDS__': error_cards(rows),
        '__RATING_ROWS__': rating_rows(stats),
    }
    for token, value in replacements.items():
        document = document.replace(token, value)
    if any(token in document for token in replacements):
        raise ValueError('Dashboard template replacement failed')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(document, encoding='utf-8')
    print(f'Built {OUTPUT_PATH} from {len(rows)} CSV rows.')


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="MBAX 6418 sentiment-classification evaluation dashboard for 100 Amazon Gift Card reviews.">
  <title>Sentiment Evaluation · MBAX 6418</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230d2546'/%3E%3Cpath d='M17 43V29h8v14zm11 0V20h8v23zm11 0V12h8v31z' fill='%2320a58f'/%3E%3C/svg%3E">
  <style>
    :root {
      /* Theme controls */
      --canvas: #f3f6f8;
      --surface: #ffffff;
      --surface-soft: #f8fafb;
      --ink: #102033;
      --ink-soft: #526173;
      --navy: #0d2546;
      --positive: #137a6b;
      --positive-soft: #e4f3ef;
      --negative: #b33c4a;
      --negative-soft: #fbeaec;
      --caution: #b56c16;
      --caution-soft: #fff3df;
      --line: #dbe2e8;
      --shadow: 0 12px 32px rgba(20, 38, 58, .07);
      --radius: 16px;
      --max-width: 1180px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--canvas);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 16px;
      line-height: 1.55;
    }
    button, input, select { font: inherit; }
    a { color: inherit; }
    .skip-link { position: fixed; left: 1rem; top: -5rem; z-index: 20; padding: .7rem 1rem; background: white; border-radius: 8px; }
    .skip-link:focus { top: 1rem; }
    .site-header {
      background: var(--navy);
      color: white;
      position: relative;
      overflow: hidden;
    }
    .site-header::after {
      content: "";
      position: absolute;
      width: 30rem;
      height: 30rem;
      right: -10rem;
      top: -17rem;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 50%;
      box-shadow: 0 0 0 5rem rgba(255,255,255,.025), 0 0 0 10rem rgba(255,255,255,.02);
    }
    .header-inner, main, .footer-inner { width: min(calc(100% - 2.5rem), var(--max-width)); margin-inline: auto; }
    .header-inner { position: relative; z-index: 1; padding: 3.25rem 0 3rem; }
    .eyebrow { margin: 0 0 .65rem; color: #8fd6ca; font-size: .8rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 0; max-width: 770px; font-size: clamp(2.3rem, 5vw, 4.5rem); line-height: 1.02; letter-spacing: -.055em; }
    .subtitle { margin: 1rem 0 0; color: #c9d6e4; font-size: 1.08rem; }
    .section-nav { display: flex; flex-wrap: wrap; gap: .5rem 1.25rem; margin-top: 2rem; }
    .section-nav a { color: #dce6ef; font-size: .88rem; text-decoration: none; border-bottom: 1px solid transparent; }
    .section-nav a:hover, .section-nav a:focus-visible { border-color: #8fd6ca; color: white; }
    main { padding: 2.25rem 0 5rem; }
    .overview-grid { display: grid; grid-template-columns: 1.35fr repeat(4, 1fr); gap: 1rem; margin-top: -4rem; position: relative; z-index: 2; }
    .metric-card { min-height: 142px; padding: 1.35rem; background: var(--surface); border: 1px solid rgba(219,226,232,.85); border-radius: var(--radius); box-shadow: var(--shadow); }
    .metric-card--lead { background: var(--positive); color: white; }
    .metric-label { display: block; color: var(--ink-soft); font-size: .82rem; font-weight: 750; letter-spacing: .04em; text-transform: uppercase; }
    .metric-card--lead .metric-label { color: #d7f2ec; }
    .metric-value { display: block; margin-top: .35rem; font-size: clamp(2rem, 3.8vw, 3rem); font-weight: 800; line-height: 1; letter-spacing: -.045em; font-variant-numeric: tabular-nums; }
    .metric-note { display: block; margin-top: .75rem; color: var(--ink-soft); font-size: .82rem; line-height: 1.35; }
    .metric-card--lead .metric-note { color: #e1f6f1; }
    section { scroll-margin-top: 1rem; margin-top: 4.5rem; }
    .section-heading { display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, 480px); gap: 2rem; align-items: end; margin-bottom: 1.5rem; }
    .section-index { display: block; margin-bottom: .35rem; color: var(--positive); font-size: .78rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
    h2 { margin: 0; font-size: clamp(1.7rem, 3vw, 2.4rem); line-height: 1.1; letter-spacing: -.035em; }
    .section-heading p { margin: 0; color: var(--ink-soft); }
    .analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
    .panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); padding: 1.6rem; }
    .panel h3 { margin: 0 0 .3rem; font-size: 1rem; }
    .panel-intro { margin: 0 0 1.5rem; color: var(--ink-soft); font-size: .9rem; }
    .balance-numbers { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
    .balance-number { padding: 1rem; border-radius: 12px; background: var(--surface-soft); }
    .balance-number span { display: block; color: var(--ink-soft); font-size: .82rem; }
    .balance-number strong { display: block; margin-top: .15rem; font-size: 2rem; letter-spacing: -.04em; }
    .balance-number--negative strong { color: var(--negative); }
    .stacked-bar { display: flex; height: 28px; overflow: hidden; border-radius: 7px; background: var(--line); }
    .stacked-bar span { display: grid; place-items: center; color: white; font-size: .72rem; font-weight: 800; }
    .stacked-positive { width: 93%; background: var(--positive); }
    .stacked-negative { width: 7%; min-width: 42px; background: var(--negative); }
    .imbalance-note { display: flex; gap: .75rem; align-items: flex-start; margin: 1.25rem 0 0; padding: 1rem; border-left: 4px solid var(--caution); background: var(--caution-soft); color: #70430d; font-size: .9rem; }
    .recall-list { display: grid; gap: 1.5rem; margin-top: 1.5rem; }
    .recall-top { display: flex; justify-content: space-between; gap: 1rem; margin-bottom: .55rem; }
    .recall-top span { font-weight: 700; }
    .recall-top strong { font-variant-numeric: tabular-nums; }
    .progress { height: 12px; overflow: hidden; border-radius: 999px; background: #e9eef1; }
    .progress i { display: block; height: 100%; border-radius: inherit; background: var(--positive); }
    .progress--negative i { background: var(--negative); }
    .recall-detail { margin: .5rem 0 0; color: var(--ink-soft); font-size: .82rem; }
    .matrix-wrap { overflow-x: auto; }
    .matrix { width: 100%; max-width: 650px; margin: 0 auto; border-collapse: separate; border-spacing: .65rem; table-layout: fixed; }
    .matrix caption { text-align: left; padding: 0 .65rem 1rem; color: var(--ink-soft); }
    .matrix th { color: var(--ink-soft); font-size: .82rem; }
    .matrix td { height: 135px; padding: 1rem; text-align: center; border-radius: 12px; }
    .matrix td strong { display: block; font-size: 2.4rem; line-height: 1; }
    .matrix td span { display: block; margin-top: .5rem; font-size: .82rem; }
    .matrix-correct { color: #075b4f; background: var(--positive-soft); border: 1px solid #abd8cf; }
    .matrix-error { color: #8e2734; background: var(--negative-soft); border: 1px solid #efbec4; }
    .axis-label { color: var(--ink); font-weight: 800; }
    .rating-panel { padding: .75rem 1.2rem 1.2rem; overflow-x: auto; }
    .rating-table { width: 100%; min-width: 700px; border-collapse: collapse; }
    .rating-table th, .rating-table td { padding: 1rem .85rem; text-align: left; border-bottom: 1px solid var(--line); }
    .rating-table thead th { color: var(--ink-soft); font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; }
    .rating-table tbody tr:last-child th, .rating-table tbody tr:last-child td { border-bottom: 0; }
    .rating-row--highlight { background: var(--caution-soft); }
    .stars { color: #d58d2c; letter-spacing: .05em; white-space: nowrap; }
    .stars span { color: #d8dee4; }
    .count-cell, .accuracy-cell { display: grid; grid-template-columns: 56px minmax(100px, 1fr); align-items: center; gap: .75rem; }
    .mini-bar { display: block; height: 8px; overflow: hidden; border-radius: 999px; background: #e9eef1; }
    .mini-bar i { display: block; height: 100%; border-radius: inherit; background: #7890a5; }
    .mini-bar--accuracy i { background: var(--positive); }
    .error-count { display: inline-grid; min-width: 2rem; height: 2rem; place-items: center; border-radius: 8px; background: var(--surface-soft); font-weight: 800; }
    .error-count.has-error { color: var(--negative); background: var(--negative-soft); }
    .rating-footnote { margin: 1rem .85rem .25rem; color: var(--ink-soft); font-size: .88rem; }
    .error-summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; overflow: hidden; margin-bottom: 1.25rem; border: 1px solid var(--line); border-radius: var(--radius); background: var(--line); }
    .error-summary > div { padding: 1.1rem 1.25rem; background: var(--surface); }
    .error-summary span { display: block; color: var(--ink-soft); font-size: .8rem; }
    .error-summary strong { display: block; margin-top: .15rem; font-size: 1.3rem; }
    .error-stack { display: grid; gap: 1.25rem; }
    .error-card { padding: 1.7rem; background: var(--surface); border: 1px solid var(--line); border-left: 5px solid var(--negative); border-radius: var(--radius); box-shadow: var(--shadow); }
    .error-card__top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
    .row-number { color: var(--negative); font-size: .78rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
    .error-card h3 { margin: .2rem 0 0; font-size: 1.25rem; }
    .error-type { flex: none; padding: .35rem .65rem; color: #793a0a; background: var(--caution-soft); border-radius: 999px; font-size: .78rem; font-weight: 750; }
    blockquote { margin: 1.25rem 0; padding: 1rem 1.15rem; background: var(--surface-soft); border-radius: 10px; color: #34475b; font-size: .94rem; }
    .label-comparison { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }
    .label-comparison > div { padding: .8rem; border: 1px solid var(--line); border-radius: 10px; }
    .label-comparison span { display: block; margin-bottom: .35rem; color: var(--ink-soft); font-size: .75rem; }
    .pill { display: inline-flex; align-items: center; min-height: 27px; padding: .25rem .55rem; border-radius: 999px; font-size: .76rem; letter-spacing: .025em; }
    .pill--positive { color: #075b4f; background: var(--positive-soft); }
    .pill--negative { color: #8e2734; background: var(--negative-soft); }
    .pill--correct { color: #075b4f; background: var(--positive-soft); }
    .pill--incorrect { color: #8e2734; background: var(--negative-soft); }
    .error-explanation { margin: 1.1rem 0 0; color: var(--ink-soft); }
    .explorer-shell { overflow: hidden; padding: 0; }
    .filters { display: grid; grid-template-columns: minmax(220px, 2fr) repeat(4, 1fr) auto; gap: .75rem; align-items: end; padding: 1.25rem; background: var(--surface-soft); border-bottom: 1px solid var(--line); }
    .control label { display: block; margin-bottom: .35rem; color: var(--ink-soft); font-size: .76rem; font-weight: 750; }
    .control input, .control select { width: 100%; height: 42px; padding: 0 .75rem; color: var(--ink); background: white; border: 1px solid #bdc8d1; border-radius: 8px; }
    .control input:focus, .control select:focus, button:focus-visible, summary:focus-visible { outline: 3px solid rgba(19,122,107,.25); outline-offset: 2px; border-color: var(--positive); }
    .clear-button { height: 42px; padding: 0 .9rem; border: 1px solid #bdc8d1; border-radius: 8px; background: white; color: var(--ink); cursor: pointer; }
    .clear-button:hover { background: #edf2f5; }
    .table-status { display: flex; justify-content: space-between; gap: 1rem; padding: .9rem 1.25rem; color: var(--ink-soft); font-size: .85rem; border-bottom: 1px solid var(--line); }
    .table-status strong { color: var(--ink); }
    .table-scroll { max-height: 680px; overflow: auto; }
    .review-table { width: 100%; min-width: 940px; border-collapse: collapse; }
    .review-table caption { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
    .review-table th, .review-table td { padding: .8rem .9rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
    .review-table thead th { position: sticky; top: 0; z-index: 2; background: #edf2f5; color: #425366; font-size: .76rem; letter-spacing: .04em; text-transform: uppercase; }
    .review-table tbody tr:hover { background: #f8fbfc; }
    .review-table tbody tr.is-error { background: #fff7f8; }
    .review-table .numeric { font-variant-numeric: tabular-nums; white-space: nowrap; }
    .review-title { max-width: 180px; font-weight: 700; }
    .review-text { min-width: 260px; max-width: 420px; }
    details summary { color: #405369; cursor: pointer; }
    details[open] summary { margin-bottom: .65rem; }
    details p { margin: 0; padding: .75rem; background: var(--surface-soft); border-radius: 8px; color: var(--ink); white-space: pre-line; }
    .empty-row td { padding: 2.5rem; text-align: center; color: var(--ink-soft); }
    .takeaways { display: grid; grid-template-columns: .8fr 1.4fr; gap: 1.25rem; }
    .takeaway-score { display: grid; align-content: space-between; min-height: 290px; color: white; background: var(--navy); }
    .takeaway-score .metric-label { color: #b8c8d8; }
    .takeaway-score strong { display: block; font-size: 4.5rem; line-height: 1; letter-spacing: -.06em; }
    .takeaway-score p { margin: 1rem 0 0; color: #cbd7e3; }
    .takeaway-list { margin: 0; padding: 0; list-style: none; }
    .takeaway-list li { position: relative; padding: 1rem 0 1rem 2rem; border-bottom: 1px solid var(--line); }
    .takeaway-list li:first-child { padding-top: .2rem; }
    .takeaway-list li:last-child { border-bottom: 0; padding-bottom: .2rem; }
    .takeaway-list li::before { content: ""; position: absolute; left: 0; top: 1.45rem; width: 9px; height: 9px; border-radius: 2px; background: var(--positive); }
    .takeaway-list li:first-child::before { top: .65rem; }
    footer { padding: 1.5rem 0; color: #647386; border-top: 1px solid var(--line); font-size: .82rem; }
    .footer-inner { display: flex; justify-content: space-between; gap: 1rem; }

    @media (max-width: 980px) {
      .overview-grid { grid-template-columns: repeat(2, 1fr); }
      .metric-card--lead { grid-column: 1 / -1; }
      .filters { grid-template-columns: repeat(2, 1fr); }
      .control--search { grid-column: 1 / -1; }
    }
    @media (max-width: 720px) {
      .header-inner, main, .footer-inner { width: min(calc(100% - 1.4rem), var(--max-width)); }
      .header-inner { padding-top: 2.3rem; }
      .overview-grid, .analysis-grid, .section-heading, .takeaways { grid-template-columns: 1fr; }
      .overview-grid { margin-top: -2rem; }
      .metric-card--lead { grid-column: auto; }
      section { margin-top: 3.5rem; }
      .section-heading { gap: .75rem; }
      .error-summary, .label-comparison { grid-template-columns: 1fr; }
      .error-card__top { display: grid; }
      .filters { grid-template-columns: 1fr; }
      .control--search { grid-column: auto; }
      .table-status, .footer-inner { display: grid; }
    }
    @media print {
      :root { --canvas: white; --shadow: none; }
      .section-nav, .filters, .clear-button { display: none; }
      .site-header { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
      .overview-grid { margin-top: 1rem; }
      .panel, .metric-card, .error-card { break-inside: avoid; }
      .table-scroll { max-height: none; overflow: visible; }
      .review-table thead th { position: static; }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to main content</a>
  <header class="site-header">
    <div class="header-inner">
      <p class="eyebrow">MBAX 6418 · Step 3</p>
      <h1>Sentiment evaluation</h1>
      <p class="subtitle">LLM sentiment predictions compared with rating-derived labels.</p>
      <nav class="section-nav" aria-label="Dashboard sections">
        <a href="#performance">Performance</a>
        <a href="#ratings">Star ratings</a>
        <a href="#errors">Error analysis</a>
        <a href="#explorer">Review explorer</a>
        <a href="#takeaways">Takeaways</a>
      </nav>
    </div>
  </header>

  <main id="main">
    <section class="overview-grid" aria-label="Evaluation overview">
      <article class="metric-card metric-card--lead"><span class="metric-label">Overall accuracy</span><strong class="metric-value">98.00%</strong><span class="metric-note">98 of 100 predictions matched the rating-derived label.</span></article>
      <article class="metric-card"><span class="metric-label">Reviews</span><strong class="metric-value">100</strong><span class="metric-note">First 100 dataset rows</span></article>
      <article class="metric-card"><span class="metric-label">Correct</span><strong class="metric-value">98</strong><span class="metric-note">Matching predictions</span></article>
      <article class="metric-card"><span class="metric-label">Incorrect</span><strong class="metric-value">2</strong><span class="metric-note">Reviewed in detail below</span></article>
      <article class="metric-card"><span class="metric-label">Balanced accuracy</span><strong class="metric-value">92.32%</strong><span class="metric-note">Average recall across both classes</span></article>
    </section>

    <section id="performance">
      <div class="section-heading">
        <div><span class="section-index">01 · Performance</span><h2>Strong results, viewed in context</h2></div>
        <p>Raw accuracy is excellent. Class balance and class-specific recall reveal the smaller negative class more clearly.</p>
      </div>
      <div class="analysis-grid">
        <article class="panel">
          <h3>Class balance</h3>
          <p class="panel-intro">Actual sentiment derived from star ratings.</p>
          <div class="balance-numbers">
            <div class="balance-number"><span>Actual POSITIVE</span><strong>93</strong></div>
            <div class="balance-number balance-number--negative"><span>Actual NEGATIVE</span><strong>7</strong></div>
          </div>
          <div class="stacked-bar" role="img" aria-label="Class balance: 93 positive reviews and 7 negative reviews">
            <span class="stacked-positive">93%</span><span class="stacked-negative">7%</span>
          </div>
          <p class="imbalance-note"><strong>Why this matters:</strong> With 93% of reviews in one class, raw accuracy can be dominated by positive reviews. Balanced accuracy gives each class equal weight.</p>
        </article>
        <article class="panel">
          <h3>Class performance</h3>
          <p class="panel-intro">Recall measures how many reviews in each actual class were identified correctly.</p>
          <div class="recall-list">
            <div><div class="recall-top"><span>POSITIVE recall</span><strong>98.92%</strong></div><div class="progress" role="progressbar" aria-label="Positive recall" aria-valuemin="0" aria-valuemax="100" aria-valuenow="98.92"><i style="width:98.92%"></i></div><p class="recall-detail">92 of 93 actual positive reviews</p></div>
            <div><div class="recall-top"><span>NEGATIVE recall</span><strong>85.71%</strong></div><div class="progress progress--negative" role="progressbar" aria-label="Negative recall" aria-valuemin="0" aria-valuemax="100" aria-valuenow="85.71"><i style="width:85.71%"></i></div><p class="recall-detail">6 of 7 actual negative reviews</p></div>
          </div>
        </article>
      </div>
    </section>

    <section id="matrix">
      <div class="section-heading">
        <div><span class="section-index">02 · Confusion matrix</span><h2>Where predictions landed</h2></div>
        <p>Green cells are correct classifications. Red cells are the two disagreements.</p>
      </div>
      <article class="panel matrix-wrap">
        <table class="matrix">
          <caption>Rows show rating-derived actual sentiment; columns show model predictions.</caption>
          <thead><tr><th></th><th colspan="2" class="axis-label">Predicted sentiment</th></tr><tr><th></th><th>POSITIVE</th><th>NEGATIVE</th></tr></thead>
          <tbody>
            <tr><th scope="row">Actual POSITIVE</th><td class="matrix-correct"><strong>92</strong><span>True positives</span></td><td class="matrix-error"><strong>1</strong><span>False negatives</span></td></tr>
            <tr><th scope="row">Actual NEGATIVE</th><td class="matrix-error"><strong>1</strong><span>False positives</span></td><td class="matrix-correct"><strong>6</strong><span>True negatives</span></td></tr>
          </tbody>
        </table>
      </article>
    </section>

    <section id="ratings">
      <div class="section-heading">
        <div><span class="section-index">03 · Star ratings</span><h2>Performance by rating</h2></div>
        <p>The sample is concentrated at 5 stars. The only 3-star error exposes a disagreement between positive wording and the assignment’s label rule.</p>
      </div>
      <article class="panel rating-panel">
        <table class="rating-table">
          <thead><tr><th>Rating</th><th>Reviews</th><th>Accuracy</th><th>Errors</th></tr></thead>
          <tbody>__RATING_ROWS__</tbody>
        </table>
        <p class="rating-footnote">Ratings of 4–5 map to POSITIVE; ratings below 4 map to NEGATIVE. The 3-star row is highlighted because one positively worded review conflicts with that rule.</p>
      </article>
    </section>

    <section id="errors">
      <div class="section-heading">
        <div><span class="section-index">04 · Error analysis</span><h2>Two disagreements, both nuanced</h2></div>
        <p>Both disagreements involve sentiment or labeling nuance rather than simple, unambiguous mistakes.</p>
      </div>
      <div class="error-summary" aria-label="Error type summary">
        <div><span>Total errors</span><strong>2</strong></div>
        <div><span>Text / rating nuance</span><strong>2</strong></div>
        <div><span>Clear-cut polarity failures</span><strong>0</strong></div>
      </div>
      <div class="error-stack">__ERROR_CARDS__</div>
    </section>

    <section id="explorer">
      <div class="section-heading">
        <div><span class="section-index">05 · Review explorer</span><h2>Inspect all 100 reviews</h2></div>
        <p>Search titles or review text, combine filters, and expand any review to read its full text.</p>
      </div>
      <article class="panel explorer-shell">
        <div class="filters" role="search">
          <div class="control control--search"><label for="search">Search reviews</label><input id="search" type="search" placeholder="Search title or review text…" autocomplete="off"></div>
          <div class="control"><label for="correct-filter">Result</label><select id="correct-filter"><option value="">All results</option><option value="true">Correct</option><option value="false">Incorrect</option></select></div>
          <div class="control"><label for="actual-filter">Actual sentiment</label><select id="actual-filter"><option value="">All actual labels</option><option value="POSITIVE">POSITIVE</option><option value="NEGATIVE">NEGATIVE</option></select></div>
          <div class="control"><label for="predicted-filter">Predicted sentiment</label><select id="predicted-filter"><option value="">All predictions</option><option value="POSITIVE">POSITIVE</option><option value="NEGATIVE">NEGATIVE</option></select></div>
          <div class="control"><label for="rating-filter">Star rating</label><select id="rating-filter"><option value="">All ratings</option><option value="5">5 stars</option><option value="4">4 stars</option><option value="3">3 stars</option><option value="2">2 stars</option><option value="1">1 star</option></select></div>
          <button class="clear-button" id="clear-filters" type="button">Clear</button>
        </div>
        <div class="table-status"><span id="result-count" aria-live="polite"><strong>100</strong> of 100 reviews</span><span>Full review text is embedded in this offline file.</span></div>
        <div class="table-scroll">
          <table class="review-table">
            <caption>All evaluated reviews and their sentiment results</caption>
            <thead><tr><th>Row</th><th>Rating</th><th>Title</th><th>Review text</th><th>Actual</th><th>Predicted</th><th>Result</th></tr></thead>
            <tbody id="review-rows"></tbody>
          </table>
        </div>
        <noscript><p class="empty-row">JavaScript is required to display and filter the embedded review table.</p></noscript>
      </article>
    </section>

    <section id="takeaways">
      <div class="section-heading">
        <div><span class="section-index">06 · Interpretation</span><h2>What these results suggest</h2></div>
        <p>Conclusions apply to this first 100-review evaluation and its rating-derived labels.</p>
      </div>
      <div class="takeaways">
        <article class="panel takeaway-score"><div><span class="metric-label">Overall performance</span><strong>98%</strong></div><p>Very strong agreement across the 100 evaluated reviews.</p></article>
        <article class="panel"><ul class="takeaway-list">
          <li>The sample is highly imbalanced: 93 actual POSITIVE reviews versus 7 actual NEGATIVE reviews.</li>
          <li>Balanced accuracy of 92.32% gives a more cautious view than the 98.00% raw accuracy because it weighs both classes equally.</li>
          <li>The model performed strongly on both classes, with 98.92% POSITIVE recall and 85.71% NEGATIVE recall. The negative estimate is based on only seven examples.</li>
          <li>Row 18 mixes a positive overall rating with a substantial complaint, creating a legitimate sentiment-interpretation challenge.</li>
          <li>Row 99 uses positive language, but the assignment’s rule converts its 3-star rating to NEGATIVE. This is a label-rule mismatch rather than an obvious language failure.</li>
        </ul></article>
      </div>
    </section>
  </main>

  <footer><div class="footer-inner"><span>MBAX 6418 · Amazon Gift Card Review Sentiment</span><span>Source: first 100 rows in predictions.csv</span></div></footer>

  <script id="dashboard-data" type="application/json">__DATA_JSON__</script>
  <script>
    (() => {
      'use strict';
      const reviews = JSON.parse(document.getElementById('dashboard-data').textContent);
      const controls = {
        search: document.getElementById('search'),
        correct: document.getElementById('correct-filter'),
        actual: document.getElementById('actual-filter'),
        predicted: document.getElementById('predicted-filter'),
        rating: document.getElementById('rating-filter')
      };
      const tbody = document.getElementById('review-rows');
      const resultCount = document.getElementById('result-count');

      const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
      })[character]);

      const shorten = (text, limit = 115) => {
        const clean = normalizeText(text).replace(/\s+/g, ' ').trim();
        if (clean.length <= limit) return clean;
        const slice = clean.slice(0, limit - 1);
        return slice.slice(0, Math.max(slice.lastIndexOf(' '), limit - 18)) + '…';
      };

      const normalizeText = (text) => text
        .replace(/<br\s*\/?>/gi, '\n')
        .replace(/&#34;|&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&amp;/g, '&');

      function filterReviews(filters) {
        const query = (filters.search || '').trim().toLocaleLowerCase();
        return reviews.filter(review => {
          const matchesSearch = !query || `${review.title} ${review.text}`.toLocaleLowerCase().includes(query);
          const matchesCorrect = filters.correct === '' || String(review.correct) === filters.correct;
          const matchesActual = !filters.actual || review.actual_sentiment === filters.actual;
          const matchesPredicted = !filters.predicted || review.predicted_sentiment === filters.predicted;
          const matchesRating = !filters.rating || String(review.rating) === filters.rating;
          return matchesSearch && matchesCorrect && matchesActual && matchesPredicted && matchesRating;
        });
      }

      function currentFilters() {
        return Object.fromEntries(Object.entries(controls).map(([key, control]) => [key, control.value]));
      }

      function render() {
        const visible = filterReviews(currentFilters());
        resultCount.innerHTML = `<strong>${visible.length}</strong> of ${reviews.length} reviews`;
        if (!visible.length) {
          tbody.innerHTML = '<tr class="empty-row"><td colspan="7">No reviews match these filters.</td></tr>';
          return;
        }
        tbody.innerHTML = visible.map(review => {
          const actualClass = review.actual_sentiment === 'POSITIVE' ? 'positive' : 'negative';
          const predictedClass = review.predicted_sentiment === 'POSITIVE' ? 'positive' : 'negative';
          return `<tr class="${review.correct ? '' : 'is-error'}">
            <td class="numeric">${review.row_id}</td>
            <td class="numeric"><span class="stars" aria-label="${review.rating} stars">${'★'.repeat(review.rating)}<span>${'★'.repeat(5-review.rating)}</span></span></td>
            <td class="review-title">${escapeHtml(review.title)}</td>
            <td class="review-text"><details><summary>${escapeHtml(shorten(review.text))}</summary><p>${escapeHtml(normalizeText(review.text))}</p></details></td>
            <td><span class="pill pill--${actualClass}">${review.actual_sentiment}</span></td>
            <td><span class="pill pill--${predictedClass}">${review.predicted_sentiment}</span></td>
            <td><span class="pill pill--${review.correct ? 'correct' : 'incorrect'}">${review.correct ? 'Correct' : 'Incorrect'}</span></td>
          </tr>`;
        }).join('');
      }

      Object.values(controls).forEach(control => control.addEventListener('input', render));
      document.getElementById('clear-filters').addEventListener('click', () => {
        Object.values(controls).forEach(control => { control.value = ''; });
        render();
        controls.search.focus();
      });

      window.__dashboardTest = { reviews, filterReviews };
      render();
    })();
  </script>
</body>
</html>
'''


if __name__ == '__main__':
    build()
