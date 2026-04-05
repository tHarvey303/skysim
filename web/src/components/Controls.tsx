import React from "react";
import type { TelescopeInfo } from "../api";

export type RenderMode = "single" | "rgb";

export interface ControlValues {
  ra: number;
  dec: number;
  seed: number;
  telescope: string;
  nside: number;
  fovArcmin: number;
  exposureTimeS: number;
  magLimit: number;
  psfFwhm: number;
  psfType: string;
  includeStars: boolean;
  mode: RenderMode;
  filter: string;
  filterR: string;
  filterG: string;
  filterB: string;
}

interface Props {
  values: ControlValues;
  onChange: (v: ControlValues) => void;
  onRender: () => void;
  onPan?: (direction: "N" | "S" | "E" | "W") => void;
  filters: string[];
  telescopes: Record<string, TelescopeInfo>;
  loading: boolean;
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(true);
  return (
    <div className="section">
      <div className="section-header" onClick={() => setOpen(!open)}>
        {title}
        <span>{open ? "\u25B4" : "\u25BE"}</span>
      </div>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
}

export default function Controls({
  values,
  onChange,
  onRender,
  onPan,
  filters,
  telescopes,
  loading,
}: Props) {
  const set = <K extends keyof ControlValues>(
    key: K,
    val: ControlValues[K]
  ) => {
    onChange({ ...values, [key]: val });
  };

  const telKeys = Object.keys(telescopes);

  return (
    <div className="sidebar">
      <Section title="Telescope">
        <div className="field">
          <label>Preset</label>
          <select
            value={values.telescope}
            onChange={(e) => {
              const key = e.target.value;
              set("telescope", key);
              const t = telescopes[key];
              if (t) {
                onChange({
                  ...values,
                  telescope: key,
                  fovArcmin: t.fov_arcmin,
                  exposureTimeS: t.exposure_time_s,
                });
              }
            }}
          >
            {telKeys.map((k) => (
              <option key={k} value={k}>
                {telescopes[k]?.name || k}
              </option>
            ))}
          </select>
        </div>
        <div className="field-row">
          <div className="field">
            <label>FoV (arcmin)</label>
            <input
              type="number"
              step="0.1"
              min="0.1"
              value={values.fovArcmin}
              onChange={(e) => set("fovArcmin", parseFloat(e.target.value) || 0.5)}
            />
          </div>
          <div className="field">
            <label>Exp. time (s)</label>
            <input
              type="number"
              step="100"
              min="1"
              value={values.exposureTimeS}
              onChange={(e) =>
                set("exposureTimeS", parseFloat(e.target.value) || 1000)
              }
            />
          </div>
        </div>
      </Section>

      <Section title="Pointing">
        <div className="field-row">
          <div className="field">
            <label>RA (deg)</label>
            <input
              type="number"
              step="0.1"
              value={values.ra}
              onChange={(e) => set("ra", parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className="field">
            <label>Dec (deg)</label>
            <input
              type="number"
              step="0.1"
              min="-90"
              max="90"
              value={values.dec}
              onChange={(e) => set("dec", parseFloat(e.target.value) || 0)}
            />
          </div>
        </div>
        <div className="field-row">
          <div className="field">
            <label>Seed</label>
            <input
              type="number"
              value={values.seed}
              onChange={(e) => set("seed", parseInt(e.target.value) || 0)}
            />
          </div>
          <div className="field">
            <label>NSIDE</label>
            <select
              value={values.nside}
              onChange={(e) => set("nside", parseInt(e.target.value))}
            >
              {[64, 128, 256, 512, 1024].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
        </div>
      </Section>

      <Section title="Filter">
        <div className="mode-tabs">
          <button
            className={`mode-tab ${values.mode === "single" ? "active" : ""}`}
            onClick={() => set("mode", "single")}
          >
            Single Band
          </button>
          <button
            className={`mode-tab ${values.mode === "rgb" ? "active" : ""}`}
            onClick={() => set("mode", "rgb")}
          >
            RGB
          </button>
        </div>

        {values.mode === "single" ? (
          <div className="field">
            <label>Filter</label>
            <select
              value={values.filter}
              onChange={(e) => set("filter", e.target.value)}
            >
              {filters.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="rgb-filters">
            {(["R", "G", "B"] as const).map((ch) => {
              const key = `filter${ch}` as "filterR" | "filterG" | "filterB";
              const color = ch.toLowerCase() as "r" | "g" | "b";
              return (
                <div className="rgb-row" key={ch}>
                  <div className={`rgb-dot ${color}`} />
                  <select
                    value={values[key]}
                    onChange={(e) => set(key, e.target.value)}
                  >
                    {filters.map((f) => (
                      <option key={f} value={f}>
                        {f}
                      </option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>
        )}
      </Section>

      <Section title="Rendering">
        <div className="field">
          <label>Mag limit</label>
          <div className="range-row">
            <input
              type="range"
              min="20"
              max="32"
              step="0.5"
              value={values.magLimit}
              onChange={(e) => set("magLimit", parseFloat(e.target.value))}
            />
            <span className="range-value">{values.magLimit.toFixed(1)}</span>
          </div>
        </div>
        <div className="field">
          <label>PSF FWHM (arcsec)</label>
          <div className="range-row">
            <input
              type="range"
              min="0.02"
              max="1.0"
              step="0.01"
              value={values.psfFwhm}
              onChange={(e) => set("psfFwhm", parseFloat(e.target.value))}
            />
            <span className="range-value">{values.psfFwhm.toFixed(2)}</span>
          </div>
        </div>
        <div className="field-row">
          <div className="field">
            <label>PSF type</label>
            <select
              value={values.psfType}
              onChange={(e) => set("psfType", e.target.value)}
            >
              <option value="gaussian">Gaussian</option>
              <option value="moffat">Moffat</option>
              <option value="file">File (FITS)</option>
            </select>
          </div>
        </div>
        <div className="checkbox-row">
          <input
            type="checkbox"
            id="stars"
            checked={values.includeStars}
            onChange={(e) => set("includeStars", e.target.checked)}
          />
          <label htmlFor="stars">Include stars</label>
        </div>
      </Section>

      <button
        className="btn btn-primary btn-render"
        onClick={onRender}
        disabled={loading}
      >
        {loading ? "Rendering..." : "Render"}
      </button>
    </div>
  );
}
