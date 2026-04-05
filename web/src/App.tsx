import React, { useCallback, useEffect, useState } from "react";
import {
  downloadFits,
  fetchFilters,
  fetchTelescopes,
  renderDebug,
  renderRaw,
  type RawImageResult,
  type TelescopeInfo,
} from "./api";
import Controls, { type ControlValues } from "./components/Controls";
import ImageViewer from "./components/ImageViewer";
import SkyMiniMap from "./components/SkyMiniMap";

const DEFAULT_CONTROLS: ControlValues = {
  ra: 180.0,
  dec: 0.0,
  seed: 42,
  telescope: "jwst_nircam",
  nside: 256,
  fovArcmin: 1.5,
  exposureTimeS: 10000,
  magLimit: 28,
  psfFwhm: 0.1,
  psfType: "file",
  includeStars: true,
  mode: "single",
  filter: "JWST/NIRCam.F200W",
  filterR: "JWST/NIRCam.F444W",
  filterG: "JWST/NIRCam.F200W",
  filterB: "JWST/NIRCam.F090W",
  debugProperty: "redshift",
};

interface Stats {
  renderTime: number;
  galaxyCount: number;
  starCount: number;
  imageSize: string;
}

export default function App() {
  const [filters, setFilters] = useState<string[]>([]);
  const [telescopes, setTelescopes] = useState<Record<string, TelescopeInfo>>(
    {}
  );
  const [controls, setControls] = useState<ControlValues>(DEFAULT_CONTROLS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);

  // Image data
  const [singleData, setSingleData] = useState<Float32Array | null>(null);
  const [rData, setRData] = useState<Float32Array | null>(null);
  const [gData, setGData] = useState<Float32Array | null>(null);
  const [bData, setBData] = useState<Float32Array | null>(null);
  const [imgWidth, setImgWidth] = useState(0);
  const [imgHeight, setImgHeight] = useState(0);

  // WCS info for cursor RA/Dec display
  const [wcsInfo, setWcsInfo] = useState<{ raCenter: number; decCenter: number; pixelScale: number } | null>(null);

  // Load filters and telescopes on mount
  useEffect(() => {
    fetchFilters().then(setFilters).catch(console.error);
    fetchTelescopes().then(setTelescopes).catch(console.error);
  }, []);

  const doRender = useCallback(async () => {
    setLoading(true);
    setError(null);

    const baseParams = {
      ra: controls.ra,
      dec: controls.dec,
      seed: controls.seed,
      telescope: controls.telescope,
      nside: controls.nside,
      fov_arcmin: controls.fovArcmin,
      exposure_time_s: controls.exposureTimeS,
      mag_limit: controls.magLimit,
      psf_fwhm: controls.psfFwhm,
      psf_type: controls.psfType,
      include_stars: controls.includeStars,
      filter_code: controls.filter, // overridden for RGB
    };

    try {
      if (controls.mode === "debug") {
        const result = await renderDebug({
          ra: controls.ra,
          dec: controls.dec,
          seed: controls.seed,
          telescope: controls.telescope,
          filter_code: controls.filter,
          nside: controls.nside,
          fov_arcmin: controls.fovArcmin,
          exposure_time_s: controls.exposureTimeS,
          mag_limit: controls.magLimit,
          property: controls.debugProperty,
        });
        setSingleData(result.data);
        setRData(null);
        setGData(null);
        setBData(null);
        setImgWidth(result.width);
        setImgHeight(result.height);
        setWcsInfo({ raCenter: result.raCenter, decCenter: result.decCenter, pixelScale: result.pixelScale });
        setStats({
          renderTime: result.renderTime,
          galaxyCount: 0,
          starCount: 0,
          imageSize: `${result.width}\u00d7${result.height}`,
        });
      } else if (controls.mode === "single") {
        const result = await renderRaw({
          ...baseParams,
          filter_code: controls.filter,
        });
        setSingleData(result.data);
        setRData(null);
        setGData(null);
        setBData(null);
        setImgWidth(result.width);
        setImgHeight(result.height);
        setWcsInfo({ raCenter: result.raCenter, decCenter: result.decCenter, pixelScale: result.pixelScale });
        setStats({
          renderTime: result.renderTime,
          galaxyCount: result.galaxyCount,
          starCount: result.starCount,
          imageSize: `${result.width}\u00d7${result.height}`,
        });
      } else {
        // RGB: render 3 filters in parallel
        const t0 = performance.now();
        const [rResult, gResult, bResult] = await Promise.all([
          renderRaw({ ...baseParams, filter_code: controls.filterR }),
          renderRaw({ ...baseParams, filter_code: controls.filterG }),
          renderRaw({ ...baseParams, filter_code: controls.filterB }),
        ]);
        const totalTime = (performance.now() - t0) / 1000;
        setSingleData(null);
        setRData(rResult.data);
        setGData(gResult.data);
        setBData(bResult.data);
        setImgWidth(rResult.width);
        setImgHeight(rResult.height);
        setWcsInfo({ raCenter: rResult.raCenter, decCenter: rResult.decCenter, pixelScale: rResult.pixelScale });
        setStats({
          renderTime: totalTime,
          galaxyCount: rResult.galaxyCount,
          starCount: rResult.starCount,
          imageSize: `${rResult.width}\u00d7${rResult.height}`,
        });
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [controls]);

  const panDirection = useCallback((direction: "N" | "S" | "E" | "W") => {
    const fovDeg = controls.fovArcmin / 60;
    const decRad = (controls.dec * Math.PI) / 180;
    let newRa = controls.ra;
    let newDec = controls.dec;
    switch (direction) {
      case "N": newDec = Math.min(90, controls.dec + fovDeg); break;
      case "S": newDec = Math.max(-90, controls.dec - fovDeg); break;
      case "E": newRa = (controls.ra - fovDeg / Math.cos(decRad) + 360) % 360; break;
      case "W": newRa = (controls.ra + fovDeg / Math.cos(decRad) + 360) % 360; break;
    }
    setControls((prev) => ({ ...prev, ra: parseFloat(newRa.toFixed(6)), dec: parseFloat(newDec.toFixed(6)) }));
  }, [controls.ra, controls.dec, controls.fovArcmin]);

  const handleDownloadFits = useCallback(async () => {
    const baseParams = {
      ra: controls.ra,
      dec: controls.dec,
      seed: controls.seed,
      telescope: controls.telescope,
      nside: controls.nside,
      fov_arcmin: controls.fovArcmin,
      exposure_time_s: controls.exposureTimeS,
      mag_limit: controls.magLimit,
      psf_fwhm: controls.psfFwhm,
      psf_type: controls.psfType,
      include_stars: controls.includeStars,
      filter_code: controls.mode === "single" ? controls.filter : controls.filterG,
    };
    try {
      await downloadFits(baseParams);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    }
  }, [controls]);

  return (
    <>
      <div className="header">
        <h1>
          <span>Sky</span>Sim
        </h1>
        <div className="status-bar">
          {stats && (
            <>
              <div className="stat">
                <span className="label">Image:</span>
                <span className="value">{stats.imageSize}</span>
              </div>
              <div className="stat">
                <span className="label">Galaxies:</span>
                <span className="value">
                  {stats.galaxyCount.toLocaleString()}
                </span>
              </div>
              <div className="stat">
                <span className="label">Stars:</span>
                <span className="value">{stats.starCount}</span>
              </div>
              <div className="stat">
                <span className="label">Render:</span>
                <span className="value">{stats.renderTime.toFixed(1)}s</span>
              </div>
            </>
          )}
          {error && (
            <span style={{ color: "var(--danger)" }}>Error: {error}</span>
          )}
        </div>
      </div>

      <div className="app">
        <Controls
          values={controls}
          onChange={setControls}
          onRender={doRender}
          onPan={panDirection}
          filters={filters}
          telescopes={telescopes}
          loading={loading}
        />

        <div style={{ flex: 1, display: "flex", flexDirection: "column", position: "relative" }}>
          {loading && (
            <div className="loading-overlay">
              <div>
                <div className="spinner" />
                <div className="loading-text">
                  {controls.mode === "rgb"
                    ? "Rendering 3 filters..."
                    : controls.mode === "debug"
                    ? `Rendering ${controls.debugProperty} map...`
                    : "Rendering..."}
                </div>
              </div>
            </div>
          )}
          <ImageViewer
            imageData={singleData}
            rData={rData}
            gData={gData}
            bData={bData}
            width={imgWidth}
            height={imgHeight}
            isRgb={controls.mode === "rgb"}
            onDownloadFits={handleDownloadFits}
            wcsInfo={wcsInfo}
          />
          <SkyMiniMap ra={controls.ra} dec={controls.dec} />
        </div>
      </div>
    </>
  );
}
