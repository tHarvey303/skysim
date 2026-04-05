import React from "react";

interface Props {
  ra: number;   // degrees [0, 360)
  dec: number;  // degrees [-90, 90]
}

/**
 * Aitoff projection: maps (RA, Dec) to (x, y) in [-2, 2] x [-1, 1].
 */
function aitoff(raDeg: number, decDeg: number): [number, number] {
  // Center on RA=180 so the map wraps naturally
  let lon = ((raDeg - 180 + 540) % 360) - 180; // [-180, 180]
  const lat = decDeg;
  const lonRad = (lon * Math.PI) / 180;
  const latRad = (lat * Math.PI) / 180;

  const alpha = Math.acos(Math.cos(latRad) * Math.cos(lonRad / 2));
  if (Math.abs(alpha) < 1e-10) return [0, 0];
  const sincAlpha = Math.sin(alpha) / alpha;
  const x = (2 * Math.cos(latRad) * Math.sin(lonRad / 2)) / sincAlpha;
  const y = Math.sin(latRad) / sincAlpha;
  return [x, y];
}

/**
 * Generate the elliptical outline of the Aitoff projection.
 */
function outlinePath(): string {
  const points: string[] = [];
  for (let i = 0; i <= 100; i++) {
    const t = (i / 100) * 2 * Math.PI;
    const x = 2 * Math.cos(t);
    const y = Math.sin(t);
    points.push(`${x},${y}`);
  }
  return `M${points.join("L")}Z`;
}

/**
 * Generate gridlines for the sky map.
 */
function gridLines(): string[] {
  const lines: string[] = [];
  // Dec lines
  for (const dec of [-60, -30, 0, 30, 60]) {
    const pts: string[] = [];
    for (let ra = 0; ra <= 360; ra += 3) {
      const [x, y] = aitoff(ra, dec);
      pts.push(`${x},${y}`);
    }
    lines.push(`M${pts.join("L")}`);
  }
  // RA lines
  for (let ra = 0; ra < 360; ra += 30) {
    const pts: string[] = [];
    for (let dec = -90; dec <= 90; dec += 3) {
      const [x, y] = aitoff(ra, dec);
      pts.push(`${x},${y}`);
    }
    lines.push(`M${pts.join("L")}`);
  }
  return lines;
}

const OUTLINE = outlinePath();
const GRID = gridLines();

export default function SkyMiniMap({ ra, dec }: Props) {
  const [px, py] = aitoff(ra, dec);

  return (
    <div className="sky-minimap" title={`RA=${ra.toFixed(2)} Dec=${dec.toFixed(2)}`}>
      <svg viewBox="-2.2 -1.2 4.4 2.4" xmlns="http://www.w3.org/2000/svg">
        {/* Background */}
        <path d={OUTLINE} fill="rgba(0,0,0,0.4)" stroke="rgba(255,255,255,0.3)" strokeWidth="0.02" />
        {/* Grid */}
        {GRID.map((d, i) => (
          <path key={i} d={d} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="0.01" />
        ))}
        {/* Galactic plane (approximate) */}
        {/* Pointing dot */}
        <circle cx={px} cy={-py} r="0.08" fill="#4fc3f7" stroke="white" strokeWidth="0.02" />
      </svg>
    </div>
  );
}
