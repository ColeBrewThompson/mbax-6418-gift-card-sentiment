"""Build a self-contained Step 7 dashboard from the balanced CSV."""
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from giftcard_reviews.balanced_evaluation import FIELDS, metrics
from giftcard_reviews.sentiment import THREE_CLASS_LABELS

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'outputs/balanced_three_class.csv'
OUTPUT = ROOT / 'dashboard/index.html'


def load_rows():
    with SOURCE.open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError('Unexpected balanced CSV header')
        rows = list(reader)
    if len(rows) != 150:
        raise ValueError(f'Expected 150 balanced reviews, got {len(rows)}')
    for row in rows:
        row['sample_id'] = int(row['sample_id'])
        row['row_id'] = int(row['row_id'])
        row['rating'] = int(float(row['rating']))
        row['correct'] = row['correct'] == 'True'
        row['verified_purchase'] = row['verified_purchase'] == 'True'
        row['helpful_vote'] = int(row['helpful_vote'])
    rows.sort(key=lambda row: row['sample_id'])
    if [row['sample_id'] for row in rows] != list(range(1, 151)):
        raise ValueError('Sample IDs are incomplete or out of order')
    if len({row['row_id'] for row in rows}) != 150:
        raise ValueError('Source row IDs are duplicated')
    if Counter(row['actual_sentiment'] for row in rows) != dict.fromkeys(THREE_CLASS_LABELS, 50):
        raise ValueError('Sample is not balanced')
    return rows


def pct(numerator, denominator):
    return f'{numerator / denominator:.2%}' if denominator else '0.00%'


def descriptive_charts(rows, result, rating_counts):
    """Build count-driven bars with labels outside the track, even for rare ratings."""
    predicted = result['predicted_counts']
    correct = Counter(row['actual_sentiment'] for row in rows if row['correct'])
    rating_bars = ''.join(
        f'<div class="chart-row" data-rating="{rating}" data-count="{rating_counts[rating]}">'
        f'<span class="chart-label">{rating} ★</span>'
        f'<span class="chart-track"><i id="rating-bar-{rating}" class="chart-fill rating-fill" '
        f'data-value="{rating_counts[rating]}" style="width:{rating_counts[rating]/50:.2%}"></i></span>'
        f'<strong class="chart-value">{rating_counts[rating]}</strong></div>'
        for rating in range(1, 6)
    )
    comparison_bars = ''.join(
        f'<div class="comparison-group" data-class="{label}"><h4>{label}</h4>'
        f'<div class="chart-row" data-series="actual" data-count="50">'
        f'<span class="chart-label">Actual</span><span class="chart-track">'
        f'<i class="chart-fill actual-fill" data-value="50" style="width:{50/75:.2%}"></i>'
        f'</span><strong class="chart-value">50</strong></div>'
        f'<div class="chart-row" data-series="predicted" data-count="{predicted[label]}">'
        f'<span class="chart-label">Model</span><span class="chart-track">'
        f'<i class="chart-fill predicted-fill {label.lower()}" '
        f'data-value="{predicted[label]}" style="width:{predicted[label]/75:.2%}"></i>'
        f'</span><strong class="chart-value">{predicted[label]}</strong></div></div>'
        for label in THREE_CLASS_LABELS
    )
    outcome_bars = ''.join(
        f'<div class="outcome-row" data-class="{label}" data-correct="{correct[label]}" '
        f'data-incorrect="{50-correct[label]}"><div class="outcome-heading">'
        f'<strong>{label}</strong><span>{correct[label]} right · '
        f'{50-correct[label]} wrong</span></div>'
        f'<div class="outcome-track" role="img" '
        f'aria-label="{label}: {correct[label]} correct and {50-correct[label]} incorrect out of 50">'
        f'<i class="outcome-correct" data-value="{correct[label]}" '
        f'style="width:{correct[label]*2:.2f}%"></i>'
        f'<i class="outcome-incorrect" data-value="{50-correct[label]}" '
        f'style="width:{(50-correct[label])*2:.2f}%"></i></div>'
        f'<small>{pct(correct[label],50)} recall</small></div>'
        for label in THREE_CLASS_LABELS
    )
    return rating_bars, comparison_bars, outcome_bars


def build():
    rows = load_rows()
    result = metrics(rows)
    actual = result['actual_counts']
    matrix = result['matrix']
    rating_counts = Counter(row['rating'] for row in rows)
    rating_correct = Counter(row['rating'] for row in rows if row['correct'])
    neutral_predictions = Counter(row['predicted_sentiment'] for row in rows
                                  if row['actual_sentiment'] == 'NEUTRAL')
    errors = [row for row in rows if not row['correct']]
    emotion_counts = Counter(row['llm_emotion'] for row in rows)
    rating_bars, comparison_bars, outcome_bars = descriptive_charts(
        rows, result, rating_counts
    )

    class_cards = ''.join(
        f'<article class="class-card {label.lower()}"><div class="class-card-top">'
        f'<span>{label}</span><strong>{actual[label]}</strong></div>'
        f'<div class="meter"><i style="width:{result["recalls"][label]:.2%}"></i></div>'
        f'<p>Recall <strong>{result["recalls"][label]:.2%}</strong> · '
        f'{matrix[label,label]} of {actual[label]} identified</p></article>'
        for label in THREE_CLASS_LABELS
    )
    matrix_html = ''.join(
        '<tr><th scope="row">' + label + '</th>' + ''.join(
            f'<td class="{ "diagonal" if label == predicted else "miss" }">'
            f'<strong>{matrix[label,predicted]}</strong><span>'
            f'{pct(matrix[label,predicted],actual[label])} of {label.lower()}</span></td>'
            for predicted in THREE_CLASS_LABELS
        ) + '</tr>' for label in THREE_CLASS_LABELS
    )
    rating_html = ''.join(
        f'<tr class="{ "highlight" if rating == 3 else "" }"><th scope="row">'
        f'{rating} ★</th><td><div class="rating-count"><strong>{rating_counts[rating]}</strong>'
        f'<span class="track"><i style="width:{rating_counts[rating]/150:.2%}"></i></span>'
        f'</div></td><td>{pct(rating_correct[rating],rating_counts[rating])}</td>'
        f'<td>{rating_counts[rating]-rating_correct[rating]}</td></tr>'
        for rating in range(1, 6)
    )
    error_html = ''.join(
        f'<tr><td>{row["row_id"]}</td><td>{row["rating"]} ★</td>'
        f'<td><strong>{html.escape(row["title"])}</strong><details><summary>Read review</summary>'
        f'<p>{html.escape(row["text"])}</p></details></td>'
        f'<td><span class="badge {row["actual_sentiment"].lower()}">'
        f'{row["actual_sentiment"]}</span></td>'
        f'<td><span class="badge {row["predicted_sentiment"].lower()}">'
        f'{row["predicted_sentiment"]}</span></td></tr>'
        for row in errors[:20]
    )
    emotion_html = ''.join(
        f'<span><b>{html.escape(emotion)}</b> {count}</span>'
        for emotion, count in emotion_counts.most_common()
    )
    data = json.dumps(rows, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    document = TEMPLATE
    replacements = {
        '__TOTAL__': str(result['total']),
        '__CORRECT__': str(result['correct']),
        '__INCORRECT__': str(result['incorrect']),
        '__ACCURACY__': pct(result['correct'], result['total']),
        '__BALANCED__': f'{result["balanced_accuracy"]:.2%}',
        '__CLASS_CARDS__': class_cards,
        '__RATING_BARS__': rating_bars,
        '__COMPARISON_BARS__': comparison_bars,
        '__OUTCOME_BARS__': outcome_bars,
        '__MATRIX__': matrix_html,
        '__RATINGS__': rating_html,
        '__ERRORS__': error_html,
        '__ERROR_COUNT__': str(len(errors)),
        '__MORE_ERRORS__': (f'{len(errors)-20} additional errors are available in the '
                            'review explorer.' if len(errors) > 20 else ''),
        '__NEUTRAL_RECALL__': f'{result["recalls"]["NEUTRAL"]:.2%}',
        '__NEUTRAL_CORRECT__': str(matrix['NEUTRAL','NEUTRAL']),
        '__NEUTRAL_TO_POSITIVE__': str(neutral_predictions['POSITIVE']),
        '__NEUTRAL_TO_NEGATIVE__': str(neutral_predictions['NEGATIVE']),
        '__EMOTIONS__': emotion_html,
        '__DATA__': data,
    }
    for token, value in replacements.items():
        document = document.replace(token, value)
    if any(token in document for token in replacements):
        raise ValueError('Unreplaced dashboard token')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding='utf-8')
    print(f'Built {OUTPUT} from {len(rows)} balanced reviews.')


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Balanced three-class sentiment evaluation of Amazon Gift Card reviews.">
<title>Gift Card Reviews · Three-Class Evaluation</title>
<style>
:root {
  --paper:#f3f6f7;--card:#fff;--ink:#182934;--muted:#526571;--line:#d9e2e6;
  --navy:#163348;--positive:#087b70;--neutral:#976315;--negative:#a63d4a;
  --positive-bg:#e3f3ee;--neutral-bg:#fff3dc;--negative-bg:#fae8eb;
  --shadow:0 12px 32px rgba(19,45,58,.065);--radius:16px;--content:1180px;
}
*{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select{font:inherit} a{color:inherit} .wrap{width:min(calc(100% - 2.5rem),var(--content));margin:auto}
header{background:var(--navy);color:white;padding:3.2rem 0 5.5rem} .eyebrow{color:#80d3c5;font-size:.76rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase}
h1{font-size:clamp(2.4rem,5vw,4.5rem);line-height:1.04;letter-spacing:-.055em;margin:.55rem 0} header p{max-width:740px;color:#d0dce4;margin:1rem 0 0}
nav{display:flex;flex-wrap:wrap;gap:1.1rem;margin-top:2rem;font-size:.88rem;color:#d6e4e9} nav a{text-decoration:none;border-bottom:1px solid #738f9c;padding-bottom:.2rem}
main{padding-bottom:4rem}.metrics{display:grid;grid-template-columns:1.35fr repeat(3,1fr);gap:1rem;margin-top:-3.9rem;position:relative}.metric,.panel,.class-card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
.metric{min-height:155px;padding:1.25rem}.metric.featured{background:var(--positive);color:white;border-color:var(--positive)}.metric span{display:block;color:var(--muted);font-size:.78rem;font-weight:750;text-transform:uppercase;letter-spacing:.07em}.metric.featured span,.metric.featured small{color:#d6f3eb}.metric strong{display:block;font-size:clamp(2.1rem,3.6vw,3rem);line-height:1.1;letter-spacing:-.05em;margin:.65rem 0}.metric small{color:var(--muted)}
section{margin-top:4rem;scroll-margin-top:1rem}.heading{display:grid;grid-template-columns:1fr minmax(220px,420px);gap:1.2rem;align-items:end;margin-bottom:1.3rem}.heading span{font-size:.76rem;letter-spacing:.16em;text-transform:uppercase;font-weight:800;color:var(--positive)}h2{font-size:clamp(1.8rem,3vw,2.5rem);letter-spacing:-.04em;line-height:1.1;margin:.25rem 0 0}.heading p{margin:0;color:var(--muted)}
.panel{padding:1.5rem}.balance{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}.class-card{padding:1.25rem}.class-card-top{display:flex;justify-content:space-between;align-items:center;gap:1rem}.class-card-top span{font-size:.8rem;font-weight:800;letter-spacing:.07em}.class-card-top strong{font-size:2rem}.class-card p{color:var(--muted);font-size:.85rem;margin:.6rem 0 0}.meter,.track{display:block;height:11px;background:#e8eef0;border-radius:99px;overflow:hidden}.meter i,.track i{display:block;height:100%;background:var(--positive)}.neutral .meter i{background:var(--neutral)}.negative .meter i{background:var(--negative)}.balance-note{margin:1rem 0 0;color:var(--muted)}
.visual-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem}.visual-grid .wide{grid-column:1 / -1}.visual-card h3{font-size:1.08rem;margin:0}.visual-card>p{font-size:.86rem;color:var(--muted);margin:.35rem 0 1.2rem}.chart-list{display:grid;gap:.8rem}.chart-row{display:grid;grid-template-columns:58px minmax(0,1fr) 34px;gap:.7rem;align-items:center;min-width:0}.chart-label{font-size:.8rem;font-weight:700;color:var(--muted)}.chart-track{display:block;width:100%;min-width:0;height:14px;border-radius:4px;background:#e9eff1;overflow:hidden}.chart-fill{display:block;height:100%;border-radius:4px;min-width:0}.rating-fill{background:var(--navy)}.chart-value{font-size:.85rem;text-align:right;font-variant-numeric:tabular-nums}.comparison-list{display:grid;gap:1rem}.comparison-group{border-bottom:1px solid var(--line);padding-bottom:.75rem}.comparison-group:last-child{border:0;padding-bottom:0}.comparison-group h4{font-size:.78rem;letter-spacing:.05em;margin:0 0 .35rem}.comparison-group .chart-row{grid-template-columns:58px minmax(0,1fr) 34px;gap:.55rem}.comparison-group .chart-track{height:9px}.actual-fill{background:#9db0ba}.predicted-fill.positive{background:var(--positive)}.predicted-fill.neutral{background:var(--neutral)}.predicted-fill.negative{background:var(--negative)}.outcome-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1.3rem}.outcome-row{min-width:0}.outcome-heading{display:flex;justify-content:space-between;gap:.5rem;font-size:.81rem;margin-bottom:.55rem}.outcome-heading span{color:var(--muted)}.outcome-track{display:flex;width:100%;height:20px;overflow:hidden;border-radius:5px;background:#e9eff1}.outcome-track i{display:block;flex:none;height:100%}.outcome-correct{background:var(--positive)}.outcome-incorrect{background:var(--negative)}.outcome-row small{display:block;margin-top:.4rem;color:var(--muted)}.chart-legend{display:flex;gap:1rem;flex-wrap:wrap;color:var(--muted);font-size:.78rem}.chart-legend i{display:inline-block;width:10px;height:10px;margin-right:.35rem;border-radius:2px}.chart-audit{margin:1rem 0 0;color:var(--muted);font-size:.76rem}.chart-audit[data-state="error"]{color:var(--negative);font-weight:700}
.table-scroll{overflow:auto}.matrix,.rating-table,.review-table,.errors-table{border-collapse:collapse;width:100%}.matrix th,.matrix td,.rating-table th,.rating-table td,.review-table th,.review-table td,.errors-table th,.errors-table td{padding:.85rem;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}.matrix thead th,.rating-table thead th,.review-table thead th,.errors-table thead th{font-size:.76rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:#f3f7f8}.matrix td{text-align:center;min-width:135px}.matrix td strong{display:block;font-size:2rem}.matrix td span{font-size:.75rem}.matrix .diagonal{color:#08695d;background:var(--positive-bg)}.matrix .miss{color:#98323f;background:var(--negative-bg)}.matrix tbody tr:last-child>*{border-bottom:0}.axis{text-align:center!important}
.rating-table{min-width:620px}.rating-table .highlight{background:var(--neutral-bg)}.rating-count{display:grid;grid-template-columns:42px minmax(100px,1fr);align-items:center;gap:.7rem}.track{height:8px}.track i{background:#7897a3}.footnote{color:var(--muted);font-size:.85rem;margin:1rem 0 0}
.callout{border-left:4px solid var(--neutral);background:var(--neutral-bg);padding:1rem 1.2rem;border-radius:0 10px 10px 0}.callout strong{font-size:1.15rem}.callout p{margin:.4rem 0 0;color:#684b1b}.badge{display:inline-block;border-radius:99px;padding:.2rem .6rem;font-size:.72rem;font-weight:750;letter-spacing:.025em;white-space:nowrap}.badge.positive{background:var(--positive-bg);color:#08695d}.badge.neutral{background:var(--neutral-bg);color:#765015}.badge.negative{background:var(--negative-bg);color:#912e3c}.badge.correct{background:var(--positive-bg);color:#08695d}.badge.incorrect{background:var(--negative-bg);color:#912e3c}
.errors-table{min-width:760px}.errors-table details{margin-top:.4rem;font-size:.83rem;color:var(--muted)}details summary{cursor:pointer}details p{white-space:pre-line;max-width:550px;background:#f5f8f9;padding:.7rem;border-radius:8px;color:var(--ink)}
.filters{display:grid;grid-template-columns:minmax(190px,2fr) repeat(4,minmax(110px,1fr)) auto;gap:.7rem;align-items:end}.filter label{display:block;font-size:.72rem;font-weight:800;color:var(--muted);margin-bottom:.35rem}.filter input,.filter select,.clear{height:42px;width:100%;border:1px solid #bccbd1;border-radius:8px;background:white;color:var(--ink);padding:0 .65rem}.clear{width:auto;cursor:pointer}input:focus,select:focus,button:focus,summary:focus{outline:3px solid #b3ddd5;outline-offset:2px}.count{color:var(--muted);font-size:.85rem;margin:1rem 0}.review-table{min-width:1050px}.review-table th{position:sticky;top:0;z-index:1}.review-table td{font-size:.88rem}.review-table tr.is-error{background:#fff8f8}.review-table td:nth-child(3){max-width:180px;font-weight:650}.review-table td:nth-child(4){min-width:260px;max-width:360px}.review-scroll{max-height:700px;overflow:auto}.review-table details p{font-weight:400}.emotion-list{display:flex;flex-wrap:wrap;gap:.65rem}.emotion-list span{background:#edf3f5;padding:.4rem .7rem;border-radius:99px;font-size:.82rem}.takeaway{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.takeaway p{margin:.5rem 0 0;color:var(--muted)}footer{border-top:1px solid var(--line);color:var(--muted);padding:1.5rem 0;font-size:.82rem}
@media(max-width:980px){.metrics{grid-template-columns:repeat(2,1fr)}.filters{grid-template-columns:repeat(3,1fr)}.filter.search{grid-column:span 2}}@media(max-width:650px){.wrap{width:min(calc(100% - 1.3rem),var(--content))}.metrics,.balance,.heading,.takeaway,.visual-grid,.outcome-grid{grid-template-columns:1fr}.visual-grid .wide{grid-column:auto}.metrics{margin-top:-2rem}.filters{grid-template-columns:1fr 1fr}.filter.search{grid-column:span 2}.clear{width:100%}header{padding-top:2.2rem}section{margin-top:3rem}}
@media print{.filters,nav{display:none}.review-scroll{max-height:none}.metric,.panel,.class-card{box-shadow:none}body{background:white}}
</style>
</head>
<body>
<header><div class="wrap"><div class="eyebrow">MBAX 6418 / Evaluation 07</div><h1>Beyond positive or negative</h1><p>Three-class LLM sentiment predictions compared with rating-derived labels on a fixed-seed, balanced sample of Amazon Gift Card reviews.</p><nav><a href="#visuals">At a glance</a><a href="#classes">Class performance</a><a href="#matrix">Confusion matrix</a><a href="#ratings">Star ratings</a><a href="#errors">Disagreements</a><a href="#explorer">Review explorer</a></nav></div></header>
<main class="wrap">
<div class="metrics" aria-label="Evaluation overview"><article class="metric featured"><span>Overall accuracy</span><strong>__ACCURACY__</strong><small>__CORRECT__ correct of __TOTAL__</small></article><article class="metric"><span>Balanced accuracy</span><strong>__BALANCED__</strong><small>Mean recall across three classes</small></article><article class="metric"><span>Reviews evaluated</span><strong>__TOTAL__</strong><small>50 sampled per rating class</small></article><article class="metric"><span>Disagreements</span><strong>__INCORRECT__</strong><small>Rating label versus text prediction</small></article></div>
<section id="visuals"><div class="heading"><div><span>01 / Visual summary</span><h2>The pattern behind the score</h2></div><p>Balanced sampling makes the neutral gap plain: the model rarely mistakes clear positive or negative reviews, but often assigns 3-star language a polarity.</p></div><div class="visual-grid"><article class="panel visual-card"><h3>Star-rating distribution</h3><p>Reviews in each rating, scaled to 50.</p><div class="chart-list" id="rating-chart">__RATING_BARS__</div></article><article class="panel visual-card"><h3>Correct answer vs model output</h3><p>Actual rating-derived counts and predicted counts, on a shared 75-review scale.</p><div class="comparison-list" id="comparison-chart">__COMPARISON_BARS__</div></article><article class="panel visual-card wide"><h3>Right and wrong within each class</h3><p>Each bar represents 50 actual reviews. Green is correctly classified; red is assigned another class.</p><div class="outcome-grid" id="outcome-chart">__OUTCOME_BARS__</div></article></div><p class="chart-audit" id="chart-audit" aria-live="polite">Checking chart data and visible bars…</p></section>
<section id="classes"><div class="heading"><div><span>02 / Class performance</span><h2>Equal footing for every class</h2></div><p>The source file is dominated by positive ratings. Sampling 50 from each class makes neutral and negative behavior visible.</p></div><div class="balance">__CLASS_CARDS__</div><p class="balance-note">Rating rule: 4–5 ★ → POSITIVE · 3 ★ → NEUTRAL · 1–2 ★ → NEGATIVE. Each card contains 50 actual reviews.</p></section>
<section id="matrix"><div class="heading"><div><span>03 / Confusion matrix</span><h2>Where the model placed each review</h2></div><p>Rows are rating-derived answers. Columns are model predictions from title and text only. Shaded diagonal cells are correct.</p></div><article class="panel table-scroll"><table class="matrix"><thead><tr><th scope="col">Actual ↓ / Predicted →</th><th>POSITIVE</th><th>NEUTRAL</th><th>NEGATIVE</th></tr></thead><tbody>__MATRIX__</tbody></table></article></section>
<section id="ratings"><div class="heading"><div><span>04 / Star ratings</span><h2>A separate home for 3-star reviews</h2></div><p>The 3-star row is its own NEUTRAL class; it is no longer counted as NEGATIVE.</p></div><article class="panel table-scroll"><table class="rating-table"><thead><tr><th>Rating</th><th>Reviews</th><th>Accuracy</th><th>Errors</th></tr></thead><tbody>__RATINGS__</tbody></table><p class="footnote">Review-count bars show share of the full 150-review sample.</p></article><div class="callout" style="margin-top:1rem"><strong>3-star diagnostic: __NEUTRAL_CORRECT__ of 50 classified NEUTRAL (__NEUTRAL_RECALL__ recall)</strong><p>The other 3-star reviews were predicted POSITIVE (__NEUTRAL_TO_POSITIVE__) or NEGATIVE (__NEUTRAL_TO_NEGATIVE__). This directly measures whether they collapse into a neighboring class.</p></div></section>
<section id="errors"><div class="heading"><div><span>05 / Disagreements</span><h2>Read the exceptions</h2></div><p>__ERROR_COUNT__ predictions differ from the rating rule. The first 20 are shown here; the explorer includes every review.</p></div><article class="panel table-scroll"><table class="errors-table"><thead><tr><th>Source row</th><th>Rating</th><th>Review</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>__ERRORS__</tbody></table><p class="footnote">__MORE_ERRORS__</p></article></section>
<section id="explorer"><div class="heading"><div><span>06 / Review explorer</span><h2>Inspect all 150 sampled reviews</h2></div><p>Combine filters to inspect a class, an error type, or a star rating. Source row numbers refer to the full JSONL file.</p></div><article class="panel"><div class="filters" role="search"><div class="filter search"><label for="search">Search title or text</label><input id="search" type="search" placeholder="Find a review…"></div><div class="filter"><label for="result-filter">Result</label><select id="result-filter"><option value="">All</option><option value="true">Correct</option><option value="false">Incorrect</option></select></div><div class="filter"><label for="actual-filter">Actual</label><select id="actual-filter"><option value="">All classes</option><option>POSITIVE</option><option>NEUTRAL</option><option>NEGATIVE</option></select></div><div class="filter"><label for="predicted-filter">Predicted</label><select id="predicted-filter"><option value="">All classes</option><option>POSITIVE</option><option>NEUTRAL</option><option>NEGATIVE</option></select></div><div class="filter"><label for="rating-filter">Rating</label><select id="rating-filter"><option value="">All stars</option><option value="1">1 ★</option><option value="2">2 ★</option><option value="3">3 ★</option><option value="4">4 ★</option><option value="5">5 ★</option></select></div><button id="clear" class="clear" type="button">Clear</button></div><div class="count" id="result-count" aria-live="polite">150 of 150 reviews</div><div class="review-scroll"><table class="review-table"><thead><tr><th>Source row</th><th>Rating</th><th>Title</th><th>Review text</th><th>Actual</th><th>Predicted</th><th>Result</th><th>LLM emotion</th></tr></thead><tbody id="review-rows"></tbody></table></div></article></section>
<section id="interpretation"><div class="heading"><div><span>07 / Interpretation</span><h2>What balance reveals</h2></div><p>The original first-100 run used two classes and contained 93 positive reviews. Its 98% accuracy is not directly comparable to this new three-class, balanced task.</p></div><div class="takeaway"><article class="panel"><h3>Neutral reviews are the key test</h3><p>Of 50 actual NEUTRAL reviews, __NEUTRAL_CORRECT__ were predicted NEUTRAL. __NEUTRAL_TO_POSITIVE__ went to POSITIVE and __NEUTRAL_TO_NEGATIVE__ went to NEGATIVE. The model sees only language, while the answer key uses the rating.</p></article><article class="panel"><h3>Emotion output retained</h3><p>The same model response also includes one primary emotion for every review. LLM emotion counts in this sample:</p><div class="emotion-list">__EMOTIONS__</div></article></div></section>
</main><footer><div class="wrap">MBAX 6418 · Fixed seed 6418 · 50 reviews per rating class · Local, offline dashboard</div></footer>
<script id="dashboard-data" type="application/json">__DATA__</script>
<script>
(()=>{'use strict';const reviews=JSON.parse(document.getElementById('dashboard-data').textContent);const controls={search:document.getElementById('search'),correct:document.getElementById('result-filter'),actual:document.getElementById('actual-filter'),predicted:document.getElementById('predicted-filter'),rating:document.getElementById('rating-filter')};const tbody=document.getElementById('review-rows');const count=document.getElementById('result-count');const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const short=x=>x.length>110?x.slice(0,107)+'…':x;function filterReviews(f){const q=(f.search||'').trim().toLocaleLowerCase();return reviews.filter(r=>(!q||(r.title+' '+r.text).toLocaleLowerCase().includes(q))&&(!f.correct||String(r.correct)===f.correct)&&(!f.actual||r.actual_sentiment===f.actual)&&(!f.predicted||r.predicted_sentiment===f.predicted)&&(!f.rating||String(r.rating)===f.rating))}function render(){const f=Object.fromEntries(Object.entries(controls).map(([k,v])=>[k,v.value]));const found=filterReviews(f);count.textContent=found.length+' of '+reviews.length+' reviews';tbody.innerHTML=found.length?found.map(r=>`<tr class="${r.correct?'':'is-error'}"><td>${r.row_id}</td><td>${r.rating} ★</td><td>${esc(r.title)}</td><td><details><summary>${esc(short(r.text))}</summary><p>${esc(r.text)}</p></details></td><td><span class="badge ${r.actual_sentiment.toLowerCase()}">${r.actual_sentiment}</span></td><td><span class="badge ${r.predicted_sentiment.toLowerCase()}">${r.predicted_sentiment}</span></td><td><span class="badge ${r.correct?'correct':'incorrect'}">${r.correct?'Correct':'Incorrect'}</span></td><td>${esc(r.llm_emotion)}</td></tr>`).join(''):'<tr><td colspan="8">No reviews match these filters.</td></tr>'}Object.values(controls).forEach(c=>c.addEventListener('input',render));document.getElementById('clear').addEventListener('click',()=>{Object.values(controls).forEach(c=>c.value='');render()});window.__dashboardTest={reviews,filterReviews};render();
  // Browser-side audit: checks displayed counts against the embedded reviews
  // and checks that every nonzero chart segment occupies visible pixels.
  if (typeof document.querySelectorAll === 'function' && typeof window.addEventListener === 'function') {
    window.addEventListener('load', () => window.requestAnimationFrame(() => {
      const status = document.getElementById('chart-audit');
      const labels = ['POSITIVE','NEUTRAL','NEGATIVE'];
      const countIf = test => reviews.filter(test).length;
      const failures = [];
      for (const row of document.querySelectorAll('#rating-chart .chart-row')) {
        const expected = countIf(review => review.rating === Number(row.dataset.rating));
        if (Number(row.dataset.count) !== expected ||
            Number(row.querySelector('.chart-value').textContent) !== expected) failures.push('rating '+row.dataset.rating);
      }
      for (const label of labels) {
        const group = document.querySelector('#comparison-chart [data-class="'+label+'"]');
        for (const series of ['actual','predicted']) {
          const row = group.querySelector('[data-series="'+series+'"]');
          const expected = countIf(review => review[(series === 'actual' ? 'actual_sentiment' : 'predicted_sentiment')] === label);
          if (Number(row.dataset.count) !== expected ||
              Number(row.querySelector('.chart-value').textContent) !== expected) failures.push(label+' '+series);
        }
        const outcome = document.querySelector('#outcome-chart [data-class="'+label+'"]');
        const right = countIf(review => review.actual_sentiment === label && review.correct);
        const wrong = countIf(review => review.actual_sentiment === label && !review.correct);
        if (Number(outcome.dataset.correct) !== right || Number(outcome.dataset.incorrect) !== wrong || right + wrong !== 50) failures.push(label+' outcome');
      }
      for (const bar of document.querySelectorAll('#visuals .chart-fill, #visuals .outcome-track i')) {
        if (Number(bar.dataset.value) > 0 && bar.getBoundingClientRect().width < 1) failures.push('collapsed bar');
      }
      status.dataset.state = failures.length ? 'error' : 'pass';
      status.textContent = failures.length
        ? 'Chart verification failed: '+failures.join(', ')
        : 'Verified in browser: chart counts match all 150 embedded reviews; every nonzero bar is visible.';
      window.__chartAudit = { passed: failures.length === 0, failures };
    }));
  }
})();
</script>
</body></html>'''


if __name__ == '__main__':
    build()
