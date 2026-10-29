/* Display-only operations: never normalize each model independently. */
(function (root) {
  'use strict';
  function decode(payload) {
    if (!payload || payload.encoding !== 'float32-le-base64') throw new Error('Raw attribution is unavailable; rerun attribute_waterbirds.py.');
    const [height, width] = payload.shape;
    if (!Number.isInteger(height) || !Number.isInteger(width) || height < 1 || width < 1) throw new Error('Invalid attribution shape');
    const bytes = Uint8Array.from(atob(payload.data), c => c.charCodeAt(0));
    if (bytes.length !== height * width * 4) throw new Error('Attribution byte count does not match shape');
    const view = new DataView(bytes.buffer), values = new Float32Array(height * width);
    for (let i = 0; i < values.length; i++) {
      values[i] = view.getFloat32(i * 4, true);
      if (!Number.isFinite(values[i])) throw new Error('Nonfinite attribution');
    }
    return {values, height, width};
  }
  function aggregate(map, block) {
    if (!Number.isInteger(block) || block < 1 || map.height % block || map.width % block) throw new Error('Region size must divide the image');
    const height = map.height / block, width = map.width / block, values = new Float64Array(height * width);
    for (let y = 0; y < map.height; y++) for (let x = 0; x < map.width; x++) {
      values[Math.floor(y / block) * width + Math.floor(x / block)] += map.values[y * map.width + x];
    }
    return {values, height, width};
  }
  function scaleFor(maps, percentile = 99) {
    const values = maps.flatMap(map => Array.from(map.values, Math.abs)).sort((a, b) => a - b);
    if (!values.length || percentile <= 0 || percentile > 100) throw new Error('Invalid shared scale');
    const position = (values.length - 1) * percentile / 100, low = Math.floor(position);
    const quantile = values[low] + (values[Math.ceil(position)] - values[low]) * (position - low);
    // Sparse maps may have a zero percentile despite a few nonzero pixels.
    return quantile || values[values.length - 1];
  }
  function pixels(map, scale, mode, opacity, crop) {
    if (!['heatmap', 'overlay'].includes(mode) || !Number.isFinite(scale) || scale < 0 || opacity < 0 || opacity > 1) throw new Error('Invalid rendering options');
    const width = crop.width, height = crop.height, output = new Uint8ClampedArray(width * height * 4);
    if (width % map.width || height % map.height) throw new Error('Crop and attribution dimensions differ');
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const value = map.values[Math.floor(y * map.height / height) * map.width + Math.floor(x * map.width / width)];
      const strength = scale > 0 ? Math.min(1, Math.abs(value) / scale) : 0;
      const color = value >= 0 ? [213, 65, 21] : [27, 87, 190];
      const offset = (y * width + x) * 4;
      const gray = .2126 * crop.data[offset] + .7152 * crop.data[offset + 1] + .0722 * crop.data[offset + 2];
      // A quiet monochrome photo avoids mistaking natural colors for attribution.
      const background = mode === 'heatmap' ? 248 : 95 + .55 * gray;
      const alpha = strength * (mode === 'heatmap' ? 1 : opacity);
      for (let channel = 0; channel < 3; channel++) output[offset + channel] = background * (1 - alpha) + color[channel] * alpha;
      output[offset + 3] = 255;
    }
    return output;
  }
  function clippedFraction(map, scale) {
    return Array.from(map.values).filter(value => Math.abs(value) > scale).length / map.values.length;
  }
  const api = {decode, aggregate, scaleFor, pixels, clippedFraction};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AttributionView = api;
})(globalThis);
