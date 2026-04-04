import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  stretchToImageData,
  rgbToImageData,
  type StretchType,
  type ColormapType,
} from "../utils/stretch";

interface Props {
  /** Single-band image data (mutually exclusive with rgb*). */
  imageData: Float32Array | null;
  /** RGB channel data. */
  rData: Float32Array | null;
  gData: Float32Array | null;
  bData: Float32Array | null;
  width: number;
  height: number;
  isRgb: boolean;
}

export default function ImageViewer({
  imageData,
  rData,
  gData,
  bData,
  width,
  height,
  isRgb,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  // Pan/zoom state
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ startX: number; startY: number; ox: number; oy: number } | null>(null);

  // Stretch state
  const [stretch, setStretch] = useState<StretchType>("asinh");
  const [pmin, setPmin] = useState(0.5);
  const [pmax, setPmax] = useState(99.5);
  const [colormap, setColormap] = useState<ColormapType>("grayscale");
  const [invert, setInvert] = useState(false);

  // Cursor info
  const [cursorInfo, setCursorInfo] = useState("");

  // Render stretched image to offscreen canvas when data or stretch changes
  useEffect(() => {
    if (width === 0 || height === 0) return;

    let imgData: ImageData | null = null;
    if (isRgb && rData && gData && bData) {
      imgData = rgbToImageData(rData, gData, bData, width, height, stretch, pmin, pmax, invert);
    } else if (imageData) {
      imgData = stretchToImageData(imageData, width, height, stretch, pmin, pmax, colormap, invert);
    }
    if (!imgData) return;

    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement("canvas");
    }
    const off = offscreenRef.current;
    off.width = width;
    off.height = height;
    const ctx = off.getContext("2d")!;
    ctx.putImageData(imgData, 0, 0);
  }, [imageData, rData, gData, bData, width, height, stretch, pmin, pmax, colormap, invert, isRgb]);

  // Draw offscreen canvas to visible canvas with pan/zoom
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const off = offscreenRef.current;
    if (!canvas || !container || !off || width === 0) return;

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    const ctx = canvas.getContext("2d")!;

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(zoom, zoom);
    ctx.imageSmoothingEnabled = zoom < 4;
    ctx.drawImage(off, 0, 0);
    ctx.restore();
  }, [offset, zoom, width]);

  useEffect(() => {
    draw();
  }, [draw, imageData, rData, gData, bData, stretch, pmin, pmax, colormap, invert]);

  // Fit image on first load or when image changes
  useEffect(() => {
    const container = containerRef.current;
    if (!container || width === 0) return;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const scale = Math.min(cw / width, ch / height, 1);
    setZoom(scale);
    setOffset({
      x: (cw - width * scale) / 2,
      y: (ch - height * scale) / 2,
    });
  }, [width, height]);

  // Resize handler
  useEffect(() => {
    const obs = new ResizeObserver(() => draw());
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, [draw]);

  // Mouse handlers for pan
  const onMouseDown = (e: React.MouseEvent) => {
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      ox: offset.x,
      oy: offset.y,
    };
  };

  const onMouseMove = (e: React.MouseEvent) => {
    // Update cursor info
    const canvas = canvasRef.current;
    if (canvas && width > 0) {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const imgX = Math.floor((mx - offset.x) / zoom);
      const imgY = Math.floor((my - offset.y) / zoom);
      if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
        const idx = imgY * width + imgX;
        const src = isRgb ? rData : imageData;
        const val = src ? src[idx] : 0;
        setCursorInfo(`(${imgX}, ${imgY})  val=${val.toExponential(2)}`);
      } else {
        setCursorInfo("");
      }
    }

    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    setOffset({ x: dragRef.current.ox + dx, y: dragRef.current.oy + dy });
  };

  const onMouseUp = () => {
    dragRef.current = null;
  };

  // Wheel zoom (centered on cursor)
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.max(0.05, Math.min(50, zoom * factor));

    // Keep point under cursor fixed
    setOffset({
      x: mx - (mx - offset.x) * (newZoom / zoom),
      y: my - (my - offset.y) * (newZoom / zoom),
    });
    setZoom(newZoom);
  };

  const hasImage = isRgb ? (rData && gData && bData) : imageData;

  return (
    <div className="main">
      <div
        className="viewer"
        ref={containerRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onWheel={onWheel}
      >
        <canvas ref={canvasRef} />
        {!hasImage && (
          <div className="empty-state">
            <div className="icon">&#9733;</div>
            <p>Configure parameters and click Render</p>
          </div>
        )}
        {cursorInfo && <div className="cursor-info">{cursorInfo}</div>}
        {hasImage && (
          <div className="viewer-overlay">
            <span className="zoom-display">
              {(zoom * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      <div className="stretch-bar">
        <label>Stretch</label>
        <select
          value={stretch}
          onChange={(e) => setStretch(e.target.value as StretchType)}
        >
          <option value="linear">Linear</option>
          <option value="sqrt">Sqrt</option>
          <option value="log">Log</option>
          <option value="asinh">Asinh</option>
          <option value="power">Power</option>
        </select>

        <label>Min %</label>
        <input
          type="range"
          min="0"
          max="10"
          step="0.1"
          value={pmin}
          onChange={(e) => setPmin(parseFloat(e.target.value))}
        />
        <span className="range-value">{pmin.toFixed(1)}</span>

        <label>Max %</label>
        <input
          type="range"
          min="90"
          max="100"
          step="0.1"
          value={pmax}
          onChange={(e) => setPmax(parseFloat(e.target.value))}
        />
        <span className="range-value">{pmax.toFixed(1)}</span>

        {!isRgb && (
          <>
            <label>Colormap</label>
            <select
              value={colormap}
              onChange={(e) => setColormap(e.target.value as ColormapType)}
            >
              <option value="grayscale">Grayscale</option>
              <option value="heat">Heat</option>
              <option value="cool">Cool</option>
              <option value="viridis">Viridis</option>
            </select>
          </>
        )}

        <label className="checkbox-row" style={{ marginLeft: 8 }}>
          <input
            type="checkbox"
            checked={invert}
            onChange={(e) => setInvert(e.target.checked)}
          />
          Invert
        </label>
      </div>
    </div>
  );
}
