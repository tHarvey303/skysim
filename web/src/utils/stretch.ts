/** Image stretch and colormap utilities. */

export type StretchType = "linear" | "sqrt" | "log" | "asinh" | "power";
export type ColormapType = "grayscale" | "heat" | "viridis" | "cool";

/**
 * Compute percentile of a Float32Array (quick select via sorting a sample).
 */
export function percentile(data: Float32Array, p: number): number {
  // Sample for speed on large images
  const maxSamples = 100_000;
  let arr: Float32Array;
  if (data.length > maxSamples) {
    arr = new Float32Array(maxSamples);
    const step = data.length / maxSamples;
    for (let i = 0; i < maxSamples; i++) {
      arr[i] = data[Math.floor(i * step)];
    }
  } else {
    arr = new Float32Array(data);
  }
  arr.sort();
  const idx = Math.min(Math.floor((p / 100) * arr.length), arr.length - 1);
  return arr[idx];
}

/**
 * Apply stretch to a single value (already normalized to [0, 1]).
 */
function stretchValue(v: number, type: StretchType): number {
  switch (type) {
    case "sqrt":
      return Math.sqrt(v);
    case "log":
      return Math.log10(1 + 9 * v);
    case "asinh":
      return Math.asinh(v * 10) / Math.asinh(10);
    case "power":
      return v * v;
    default:
      return v;
  }
}

/**
 * Apply a colormap to a [0,1] value, returning [r, g, b].
 */
function applyColormap(v: number, cmap: ColormapType): [number, number, number] {
  switch (cmap) {
    case "heat": {
      const r = Math.min(1, v * 3);
      const g = Math.min(1, Math.max(0, (v - 0.33) * 3));
      const b = Math.min(1, Math.max(0, (v - 0.66) * 3));
      return [r, g, b];
    }
    case "cool": {
      return [v * 0.5, v * 0.7, v];
    }
    case "viridis": {
      // Simplified viridis approximation
      const r = 0.267 + 0.004 * v + v * v * 1.2 - v * v * v * 0.5;
      const g = 0.004 + v * 1.4 - v * v * 0.6;
      const b = 0.329 + v * 0.7 - v * v * 1.2 + v * v * v * 0.4;
      return [
        Math.max(0, Math.min(1, r)),
        Math.max(0, Math.min(1, g)),
        Math.max(0, Math.min(1, b)),
      ];
    }
    default:
      return [v, v, v];
  }
}

/**
 * Stretch a single-band float32 image to RGBA ImageData.
 */
export function stretchToImageData(
  data: Float32Array,
  width: number,
  height: number,
  stretch: StretchType,
  pmin: number,
  pmax: number,
  colormap: ColormapType,
  invert: boolean
): ImageData {
  const vmin = percentile(data, pmin);
  const vmax = percentile(data, pmax);
  const range = vmax - vmin || 1;

  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i++) {
    let v = (data[i] - vmin) / range;
    v = Math.max(0, Math.min(1, v));
    v = stretchValue(v, stretch);
    if (invert) v = 1 - v;
    const [r, g, b] = applyColormap(v, colormap);
    const j = i * 4;
    rgba[j] = r * 255;
    rgba[j + 1] = g * 255;
    rgba[j + 2] = b * 255;
    rgba[j + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}

/**
 * Composite 3 float32 images into an RGB ImageData.
 */
export function rgbToImageData(
  rData: Float32Array,
  gData: Float32Array,
  bData: Float32Array,
  width: number,
  height: number,
  stretch: StretchType,
  pmin: number,
  pmax: number,
  invert: boolean
): ImageData {
  const rMin = percentile(rData, pmin);
  const rMax = percentile(rData, pmax);
  const gMin = percentile(gData, pmin);
  const gMax = percentile(gData, pmax);
  const bMin = percentile(bData, pmin);
  const bMax = percentile(bData, pmax);

  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < rData.length; i++) {
    let r = (rData[i] - rMin) / (rMax - rMin || 1);
    let g = (gData[i] - gMin) / (gMax - gMin || 1);
    let b = (bData[i] - bMin) / (bMax - bMin || 1);
    r = stretchValue(Math.max(0, Math.min(1, r)), stretch);
    g = stretchValue(Math.max(0, Math.min(1, g)), stretch);
    b = stretchValue(Math.max(0, Math.min(1, b)), stretch);
    if (invert) { r = 1 - r; g = 1 - g; b = 1 - b; }
    const j = i * 4;
    rgba[j] = r * 255;
    rgba[j + 1] = g * 255;
    rgba[j + 2] = b * 255;
    rgba[j + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}
