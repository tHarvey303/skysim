/** API client for SkySim backend. */

export interface RenderParams {
  ra: number;
  dec: number;
  seed: number;
  telescope: string;
  filter_code: string;
  nside: number;
  fov_arcmin?: number;
  exposure_time_s?: number;
  mag_limit: number;
  psf_fwhm: number;
  psf_type: string;
  include_stars: boolean;
}

export interface RawImageResult {
  data: Float32Array;
  width: number;
  height: number;
  renderTime: number;
  galaxyCount: number;
  starCount: number;
}

export interface TelescopeInfo {
  name: string;
  pixel_scale: number;
  fov_arcmin: number;
  aperture_m: number;
  read_noise_e: number;
  dark_current_e_s: number;
  exposure_time_s: number;
}

const BASE = "/api";

function buildQuery(params: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) {
      parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
    }
  }
  return parts.join("&");
}

export async function fetchFilters(): Promise<string[]> {
  const res = await fetch(`${BASE}/filters`);
  const data = await res.json();
  return data.filters;
}

export async function fetchTelescopes(): Promise<Record<string, TelescopeInfo>> {
  const res = await fetch(`${BASE}/telescopes`);
  return res.json();
}

export async function renderRaw(params: RenderParams): Promise<RawImageResult> {
  const qs = buildQuery(params as unknown as Record<string, unknown>);
  const res = await fetch(`${BASE}/render/raw?${qs}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Render failed: ${res.status} ${text}`);
  }
  const width = parseInt(res.headers.get("X-Image-Width") || "0", 10);
  const height = parseInt(res.headers.get("X-Image-Height") || "0", 10);
  const renderTime = parseFloat(res.headers.get("X-Render-Time") || "0");
  const galaxyCount = parseInt(res.headers.get("X-Galaxy-Count") || "0", 10);
  const starCount = parseInt(res.headers.get("X-Star-Count") || "0", 10);

  const buf = await res.arrayBuffer();
  const data = new Float32Array(buf);

  return { data, width, height, renderTime, galaxyCount, starCount };
}
