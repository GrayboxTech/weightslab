const {test} = require('node:test');
const assert = require('node:assert/strict');
const view = require('./attribution_view.js');
const map = (values, width = values.length) => ({values, width, height: values.length / width});
const crop = {width: 3, height: 1, data: new Uint8ClampedArray(12).fill(120)};

test('signed float32 values decode without clipping or sign changes', () => {
  const bytes = Buffer.alloc(12);
  [-1.25, 0, 2.5].forEach((v, i) => bytes.writeFloatLE(v, i * 4));
  assert.deepEqual(Array.from(view.decode({encoding: 'float32-le-base64', shape: [1, 3], data: bytes.toString('base64')}).values), [-1.25, 0, 2.5]);
  assert.throws(() => view.decode(null), /unavailable/);
  assert.throws(() => view.decode({encoding: 'float32-le-base64', shape: [2, 3], data: bytes.toString('base64')}), /byte count/);
});
test('region sums preserve the signed total, including cancellation', () => {
  const result = view.aggregate(map([1, -1, 2, 2, 3, -3, 4, 4], 4), 2);
  assert.deepEqual(Array.from(result.values), [0, 12]);
  assert.equal(result.values.reduce((a, b) => a + b, 0), 12);
  assert.throws(() => view.aggregate(map([1, 2, 3]), 2), /divide/);
});
test('the same signed magnitude has the same color across branches', () => {
  const first = map([-1, 0, 1]), second = map([-1, 0, 4]);
  const scale = view.scaleFor([first, second], 100);
  assert.equal(scale, 4);
  const a = view.pixels(first, scale, 'heatmap', 1, crop), b = view.pixels(second, scale, 'heatmap', 1, crop);
  assert.deepEqual(a.slice(0, 8), b.slice(0, 8));
  assert.ok(a[2] > a[0]); // negative is blue
  assert.ok(a[8] > a[10]); // positive is orange
});
test('pooled percentile is symmetric, robust and clipping is explicit', () => {
  const a = map([0, 1, 2, 100]), b = map([-1, -2, -3, -100]);
  assert.equal(view.scaleFor([a, b], 50), 2);
  assert.equal(view.scaleFor([b, a], 50), 2);
  assert.equal(view.clippedFraction(a, 2), .25);
});
test('zero maps remain neutral and opacity only changes overlay', () => {
  const zeros = view.pixels(map([0, 0, 0]), 0, 'heatmap', 1, crop);
  assert.deepEqual(Array.from(zeros.slice(0, 4)), [248, 248, 248, 255]);
  const values = map([-1, 0, 1]);
  assert.deepEqual(view.pixels(values, 1, 'heatmap', 0, crop), view.pixels(values, 1, 'heatmap', 1, crop));
  assert.notDeepEqual(view.pixels(values, 1, 'overlay', 0, crop), view.pixels(values, 1, 'overlay', 1, crop));
});
test('a sparse nonzero map never disappears behind a zero percentile', () => {
  assert.equal(view.scaleFor([map([...Array(999).fill(0), -2])]), 2);
});
