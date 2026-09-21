import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const page = fs.readFileSync(new URL('../dashboard/index.html', import.meta.url), 'utf8');
const data = page.match(/<script id="dashboard-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(script.length, 1);

class Element {
  constructor(id) {
    this.id = id;
    this.value = '';
    this.innerHTML = '';
    this.textContent = id === 'dashboard-data' ? data : '';
    this.listeners = {};
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
}

const ids = ['dashboard-data', 'search', 'result-filter', 'actual-filter',
  'predicted-filter', 'rating-filter', 'clear', 'review-rows', 'result-count'];
const elements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
const document = { getElementById: id => elements[id] };
const window = {};
vm.runInNewContext(script[0][1], { document, window, JSON, Object, String });
const { reviews, filterReviews } = window.__dashboardTest;
const filter = overrides => filterReviews({
  search: '', correct: '', actual: '', predicted: '', rating: '', ...overrides
});
assert.equal(reviews.length, 150);
assert.equal(filter({ actual: 'POSITIVE' }).length, 50);
assert.equal(filter({ actual: 'NEUTRAL' }).length, 50);
assert.equal(filter({ actual: 'NEGATIVE' }).length, 50);
assert.equal(filter({ rating: '3' }).length, 50);
assert.equal(filter({ actual: 'NEUTRAL', rating: '3' }).length, 50);
assert.equal(filter({ correct: 'false' }).length,
  reviews.filter(r => !r.correct).length);
assert.equal(filter({ actual: 'NEUTRAL', predicted: 'NEUTRAL' }).length,
  reviews.filter(r => r.actual_sentiment === 'NEUTRAL' &&
    r.predicted_sentiment === 'NEUTRAL').length);
assert.equal((elements['review-rows'].innerHTML.match(/<details>/g) || []).length, 150);
elements['actual-filter'].value = 'NEUTRAL';
elements['actual-filter'].listeners.input();
assert.equal(elements['result-count'].textContent, '50 of 150 reviews');
elements['result-filter'].value = 'false';
elements['result-filter'].listeners.input();
assert.equal(Number(elements['result-count'].textContent.split(' ')[0]),
  filter({ actual: 'NEUTRAL', correct: 'false' }).length);
elements.clear.listeners.click();
assert.equal(elements['result-count'].textContent, '150 of 150 reviews');
assert.equal(elements['actual-filter'].value, '');
console.log('PASS: live filters and counts reflect exactly the requested subsets');
console.log('PASS: clear filters restores all 150 expandable reviews');
