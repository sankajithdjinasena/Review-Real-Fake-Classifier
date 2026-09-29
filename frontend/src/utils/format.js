/**
 * Smart percentage formatter that displays exact probabilities accurately without rounding
 * extreme edge-case values to misleading 100.0% or 0.0%.
 */
export function formatPercent(prob) {
  if (prob === undefined || prob === null || isNaN(prob)) return '0%';

  const val = prob <= 1 ? prob * 100 : prob;

  if (val === 0) return '0%';
  if (val === 100) return '100%';

  // If extremely close to 100% (e.g. 99.999% -> ">99.99%")
  if (val > 99.99 && val < 100) {
    return '>99.99%';
  }
  // If very close to 100% (e.g. 99.986% -> "99.99%")
  if (val > 99.9 && val <= 99.99) {
    return `${val.toFixed(2)}%`;
  }

  // If extremely close to 0% (e.g. 0.001% -> "<0.01%")
  if (val > 0 && val < 0.01) {
    return '<0.01%';
  }
  // If very close to 0% (e.g. 0.014% -> "0.01%")
  if (val >= 0.01 && val < 0.1) {
    return `${val.toFixed(2)}%`;
  }

  // Standard range (e.g. 85.3%)
  return `${val.toFixed(1)}%`;
}
