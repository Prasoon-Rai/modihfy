"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";

import obamaImg from "../ref-img/modih.jpeg";
const REFERENCE_IMAGE_PATH = obamaImg.src;

// ========== OBAMIFY-STYLE BOIDS PHYSICS MORPH ENGINE ==========
// Ported from the Rust obamify algorithm (Spu7Nix/obamify)
//
// Two phases:
//   Phase 1: Pixel assignment by brightness sorting (fast, runs <50ms)
//   Phase 2: Boids physics simulation with destination force, neighbor repulsion,
//            velocity alignment, wall bounce, and velocity damping

const SIDELEN = 256; // 256x256 = 65536 particles (sharper image)
const PERSONAL_SPACE = 0.95;
const MAX_VELOCITY = 6.0;
const ALIGNMENT_FACTOR = 0.8;
const DAMPING = 0.97;
const DST_FORCE = 0.13;
const SIM_STEPS_PER_FRAME = 2;

function factorCurve(x) {
  return Math.min(x * x * x, 1000.0);
}

// ========== PIXEL ASSIGNMENT: BRIGHTNESS SORT ==========
// This is the classic "obamify" technique from CodeBullet's video:
// Sort source pixels by brightness, sort target pixels by brightness,
// match them 1:1 by rank. Bright source pixels go to bright target slots.

// Perceptual color key: combines luminance with hue information
// This prevents random purple/magenta artifacts from brightness-only matching
function colorSortKey(r, g, b) {
  const lum = r * 0.299 + g * 0.587 + b * 0.114;
  // Compute a hue-like angle mapped to 0..1
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const chroma = max - min;
  let hue = 0;
  if (chroma > 5) {
    if (max === r) hue = ((g - b) / chroma + 6) % 6;
    else if (max === g) hue = (b - r) / chroma + 2;
    else hue = (r - g) / chroma + 4;
    hue /= 6; // 0..1
  }
  // Weight: 85% luminance, 15% hue. Keeps overall brightness matching
  // while preventing wildly different hues from swapping.
  return lum + hue * 40;
}

function computeAssignments(sourcePixels, targetPixels, sidelen) {
  const n = sidelen * sidelen;

  const srcKey = new Float32Array(n);
  const tgtKey = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    srcKey[i] = colorSortKey(sourcePixels[o], sourcePixels[o+1], sourcePixels[o+2]);
    tgtKey[i] = colorSortKey(targetPixels[o], targetPixels[o+1], targetPixels[o+2]);
  }

  const srcIndices = Array.from({ length: n }, (_, i) => i);
  const tgtIndices = Array.from({ length: n }, (_, i) => i);
  srcIndices.sort((a, b) => srcKey[a] - srcKey[b]);
  tgtIndices.sort((a, b) => tgtKey[a] - tgtKey[b]);

  const assignments = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    assignments[tgtIndices[i]] = srcIndices[i];
  }

  return assignments;
}

// ========== BOIDS PHYSICS SIMULATION ==========
// Ported from obamify's morph_sim.rs

class MorphSim {
  constructor(n, sidelen, assignments) {
    this.n = n;
    this.sidelen = sidelen;
    this.gridSize = Math.sqrt(n);
    this.pixelSize = sidelen / this.gridSize;

    // Positions, velocities, accelerations
    this.posX = new Float32Array(n);
    this.posY = new Float32Array(n);
    this.dstX = new Float32Array(n);
    this.dstY = new Float32Array(n);
    this.velX = new Float32Array(n);
    this.velY = new Float32Array(n);
    this.accX = new Float32Array(n);
    this.accY = new Float32Array(n);
    this.age = new Uint32Array(n);

    const ps = this.pixelSize;
    const gs = this.gridSize;

    // Initialize source positions (each pixel starts at its grid cell center)
    for (let i = 0; i < n; i++) {
      const x = (i % gs + 0.5) * ps;
      const y = (((i / gs) | 0) + 0.5) * ps;
      this.posX[i] = x;
      this.posY[i] = y;
    }

    // Set destination positions from assignments
    // assignments[dstIdx] = srcIdx means "source pixel srcIdx should move to target slot dstIdx"
    for (let dstIdx = 0; dstIdx < n; dstIdx++) {
      const srcIdx = assignments[dstIdx];
      const dx = (dstIdx % gs + 0.5) * ps;
      const dy = (((dstIdx / gs) | 0) + 0.5) * ps;
      this.dstX[srcIdx] = dx;
      this.dstY[srcIdx] = dy;
    }
  }

  step() {
    const n = this.n;
    const gs = this.gridSize;
    const ps = this.pixelSize;
    const sl = this.sidelen;
    const personalSpace = ps * PERSONAL_SPACE;
    const halfPS = personalSpace * 0.5;

    // Build spatial grid for neighbor lookups
    const gridLen = gs * gs;
    const gridCells = new Array(gridLen);
    for (let i = 0; i < gridLen; i++) gridCells[i] = [];

    for (let i = 0; i < n; i++) {
      const gx = Math.max(0, Math.min(gs - 1, Math.floor(this.posX[i] / ps)));
      const gy = Math.max(0, Math.min(gs - 1, Math.floor(this.posY[i] / ps)));
      gridCells[(gy * gs + gx) | 0].push(i);
    }

    // Reset accelerations
    for (let i = 0; i < n; i++) {
      this.accX[i] = 0;
      this.accY[i] = 0;
    }

    // Destination attraction force with cubic ramp
    for (let i = 0; i < n; i++) {
      const elapsed = this.age[i] / 60.0;
      const factor = factorCurve(elapsed * DST_FORCE);
      const dx = this.dstX[i] - this.posX[i];
      const dy = this.dstY[i] - this.posY[i];
      const dist = Math.sqrt(dx * dx + dy * dy);
      this.accX[i] += (dx * dist * factor) / sl;
      this.accY[i] += (dy * dist * factor) / sl;
    }

    // Wall repulsion
    for (let i = 0; i < n; i++) {
      const px = this.posX[i], py = this.posY[i];
      if (px < halfPS) this.accX[i] += (halfPS - px) / halfPS;
      else if (px > sl - halfPS) this.accX[i] -= (px - (sl - halfPS)) / halfPS;
      if (py < halfPS) this.accY[i] += (halfPS - py) / halfPS;
      else if (py > sl - halfPS) this.accY[i] -= (py - (sl - halfPS)) / halfPS;
    }

    // Neighbor repulsion + velocity alignment (boids)
    for (let i = 0; i < n; i++) {
      const px = this.posX[i], py = this.posY[i];
      const col = Math.max(0, Math.min(gs - 1, (px / ps) | 0));
      const row = Math.max(0, Math.min(gs - 1, (py / ps) | 0));
      let avgVx = 0, avgVy = 0, count = 0;

      for (let ddy = -1; ddy <= 1; ddy++) {
        for (let ddx = -1; ddx <= 1; ddx++) {
          const ncol = col + ddx;
          const nrow = row + ddy;
          if (ncol < 0 || nrow < 0 || ncol >= gs || nrow >= gs) continue;
          const cell = gridCells[(nrow * gs + ncol) | 0];
          for (let k = 0; k < cell.length; k++) {
            const j = cell[k];
            if (j === i) continue;
            const ox = this.posX[j] - px;
            const oy = this.posY[j] - py;
            const d = Math.sqrt(ox * ox + oy * oy);

            let weight = 0;
            if (d > 0.001 && d < personalSpace) {
              weight = (1.0 / d) * (personalSpace - d) / personalSpace;
              this.accX[i] -= ox * weight;
              this.accY[i] -= oy * weight;
            } else if (d <= 0.001) {
              this.accX[i] += (Math.random() - 0.5) * 0.1;
              this.accY[i] += (Math.random() - 0.5) * 0.1;
              weight = 0.5;
            }

            if (weight > 0) {
              avgVx += this.velX[j] * weight;
              avgVy += this.velY[j] * weight;
              count += weight;
            }
          }
        }
      }

      if (count > 0) {
        avgVx /= count;
        avgVy /= count;
        this.accX[i] += (avgVx - this.velX[i]) * ALIGNMENT_FACTOR;
        this.accY[i] += (avgVy - this.velY[i]) * ALIGNMENT_FACTOR;
      }
    }

    // Integrate
    for (let i = 0; i < n; i++) {
      this.velX[i] = (this.velX[i] + this.accX[i]) * DAMPING;
      this.velY[i] = (this.velY[i] + this.accY[i]) * DAMPING;
      this.posX[i] += Math.max(-MAX_VELOCITY, Math.min(MAX_VELOCITY, this.velX[i]));
      this.posY[i] += Math.max(-MAX_VELOCITY, Math.min(MAX_VELOCITY, this.velY[i]));
      this.age[i]++;
    }
  }

  isSettled() {
    let settled = 0;
    const thresh = this.pixelSize * 0.5;
    for (let i = 0; i < this.n; i++) {
      const dx = this.dstX[i] - this.posX[i];
      const dy = this.dstY[i] - this.posY[i];
      if (dx * dx + dy * dy < thresh * thresh) settled++;
    }
    return settled > this.n * 0.95;
  }
}

// ========== RENDERING ==========

function runMorphAnimation(sourceImgSrc, targetImgSrc, canvas, onProgress, onStatus, onComplete) {
  const ctx = canvas.getContext('2d');
  const sidelen = SIDELEN;
  canvas.width = sidelen;
  canvas.height = sidelen;

  const loadImg = (src) => new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load: " + src));
    img.src = src;
  });

  let cancelled = false;

  const run = async () => {
    let imgA, imgB;
    try {
      [imgA, imgB] = await Promise.all([loadImg(sourceImgSrc), loadImg(targetImgSrc)]);
    } catch (e) {
      alert("Image loading error: " + e.message);
      onComplete();
      return;
    }

    const getPixels = (img) => {
      const off = document.createElement('canvas');
      off.width = sidelen;
      off.height = sidelen;
      const octx = off.getContext('2d');
      const scale = Math.max(sidelen / img.width, sidelen / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      octx.fillStyle = "#000";
      octx.fillRect(0, 0, sidelen, sidelen);
      octx.drawImage(img, (sidelen - w) / 2, (sidelen - h) / 2, w, h);
      const idata = octx.getImageData(0, 0, sidelen, sidelen);
      const rgb = new Uint8Array(sidelen * sidelen * 3);
      for (let i = 0; i < sidelen * sidelen; i++) {
        rgb[i * 3] = idata.data[i * 4];
        rgb[i * 3 + 1] = idata.data[i * 4 + 1];
        rgb[i * 3 + 2] = idata.data[i * 4 + 2];
      }
      return rgb;
    };

    onProgress(0);
    onStatus("Loading images...");

    const sourcePixels = getPixels(imgA);
    const targetPixels = getPixels(imgB);

    // Phase 1: Compute assignments via brightness sort (fast, <100ms)
    onStatus("Sorting pixels by brightness...");
    onProgress(5);
    await new Promise(r => setTimeout(r, 30));

    const assignments = computeAssignments(sourcePixels, targetPixels, sidelen);
    onProgress(15);

    if (cancelled) return;

    // Phase 2: Initialize boids simulation
    onStatus("Initializing physics simulation...");
    const sim = new MorphSim(sidelen * sidelen, sidelen, assignments);
    onProgress(20);

    const colors = sourcePixels;
    const ps = sim.pixelSize;

    // Pre-compute the clean final image from assignments (gap-free)
    const finalImageData = ctx.createImageData(sidelen, sidelen);
    const finalData = finalImageData.data;
    for (let tgtIdx = 0; tgtIdx < sidelen * sidelen; tgtIdx++) {
      const srcIdx = assignments[tgtIdx];
      const outOff = tgtIdx * 4;
      const srcOff = srcIdx * 3;
      finalData[outOff] = colors[srcOff];
      finalData[outOff + 1] = colors[srcOff + 1];
      finalData[outOff + 2] = colors[srcOff + 2];
      finalData[outOff + 3] = 255;
    }

    // Phase 3: Run simulation
    onStatus("Running boids physics...");
    const maxFrames = 900;
    let frame = 0;

    // Gap-fill helper: dilate non-empty pixels into empty neighbors
    function fillGaps(data, w) {
      const copy = new Uint8ClampedArray(data);
      for (let y = 0; y < w; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          if (data[idx + 3] === 0) {
            // Empty pixel — grab color from nearest non-empty neighbor
            for (const [nx, ny] of [[x-1,y],[x+1,y],[x,y-1],[x,y+1],[x-1,y-1],[x+1,y-1],[x-1,y+1],[x+1,y+1]]) {
              if (nx >= 0 && nx < w && ny >= 0 && ny < w) {
                const nidx = (ny * w + nx) * 4;
                if (copy[nidx + 3] === 255) {
                  data[idx] = copy[nidx];
                  data[idx+1] = copy[nidx+1];
                  data[idx+2] = copy[nidx+2];
                  data[idx+3] = 255;
                  break;
                }
              }
            }
          }
        }
      }
    }

    const renderFrame = () => {
      if (cancelled) return;

      for (let s = 0; s < SIM_STEPS_PER_FRAME; s++) {
        sim.step();
      }
      frame += SIM_STEPS_PER_FRAME;

      // Render particles at current physics positions
      const idata = ctx.createImageData(sidelen, sidelen);
      const data = idata.data;

      for (let i = 0; i < sim.n; i++) {
        const px = Math.floor(sim.posX[i] / ps);
        const py = Math.floor(sim.posY[i] / ps);
        if (px >= 0 && px < sidelen && py >= 0 && py < sidelen) {
          const outIdx = (py * sidelen + px) * 4;
          const srcIdx = i * 3;
          data[outIdx] = colors[srcIdx];
          data[outIdx + 1] = colors[srcIdx + 1];
          data[outIdx + 2] = colors[srcIdx + 2];
          data[outIdx + 3] = 255;
        }
      }

      // Fill black gaps with nearest neighbor colors (2 passes for coverage)
      fillGaps(data, sidelen);
      fillGaps(data, sidelen);

      ctx.putImageData(idata, 0, 0);

      const progress = Math.min(100, 20 + (frame / maxFrames) * 80);
      onProgress(progress);

      const settled = frame > 200 && frame % 20 === 0 && sim.isSettled();

      if (settled || frame >= maxFrames) {
        // Render the clean final image (no gaps, perfect assignment)
        ctx.putImageData(finalImageData, 0, 0);
        onProgress(100);
        onStatus("Morph complete.");
        onComplete();
        return;
      }

      requestAnimationFrame(renderFrame);
    };

    requestAnimationFrame(renderFrame);
  };

  run();
  return () => { cancelled = true; };
}


const UploadIcon = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="17 8 12 3 7 8"></polyline>
    <line x1="12" y1="3" x2="12" y2="15"></line>
  </svg>
);

const MenuIcon = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="4" y1="6" x2="20" y2="6"></line>
    <line x1="4" y1="12" x2="20" y2="12"></line>
    <line x1="4" y1="18" x2="20" y2="18"></line>
  </svg>
);

const CloseIcon = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

const SunIcon = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5"></circle>
    <line x1="12" y1="1" x2="12" y2="3"></line>
    <line x1="12" y1="21" x2="12" y2="23"></line>
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
    <line x1="1" y1="12" x2="3" y2="12"></line>
    <line x1="21" y1="12" x2="23" y2="12"></line>
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
  </svg>
);

const MoonIcon = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
  </svg>
);

const PlusIcon = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19"></line>
    <line x1="5" y1="12" x2="19" y2="12"></line>
  </svg>
);

// ========== PAGE COMPONENT ==========

export default function MorphDashboard() {
  const canvasRef = useRef(null);
  const cancelRef = useRef(null);
  const fileInputRef = useRef(null);

  const [isAnimating, setIsAnimating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);
  const [statusText, setStatusText] = useState("Awaiting input");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isDark, setIsDark] = useState(true);
  const [history, setHistory] = useState([]); // { id, thumbnail, timestamp }

  // Apply theme on mount and toggle
  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  // Set dark on mount
  useEffect(() => {
    document.documentElement.classList.add("dark");
    return () => { if (cancelRef.current) cancelRef.current(); };
  }, []);

  const handleUpload = useCallback(async (e) => {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];
    const url = URL.createObjectURL(file);
    setHasStarted(true);
    setIsAnimating(true);
    setProgress(0);
    setStatusText("Loading...");

    if (cancelRef.current) cancelRef.current();
    await new Promise(r => setTimeout(r, 50));

    cancelRef.current = runMorphAnimation(
      url,
      REFERENCE_IMAGE_PATH,
      canvasRef.current,
      setProgress,
      setStatusText,
      () => {
        setIsAnimating(false);
        // Save to history
        const thumbCanvas = document.createElement("canvas");
        thumbCanvas.width = 64;
        thumbCanvas.height = 64;
        const tctx = thumbCanvas.getContext("2d");
        tctx.drawImage(canvasRef.current, 0, 0, 64, 64);
        const thumb = thumbCanvas.toDataURL("image/jpeg", 0.7);
        setHistory(prev => [
          { id: Date.now(), thumbnail: thumb, timestamp: new Date().toLocaleTimeString() },
          ...prev
        ]);
      }
    );
    // Reset file input so same file can be re-uploaded
    e.target.value = "";
  }, []);

  const handleNewImage = () => {
    if (cancelRef.current) cancelRef.current();
    setHasStarted(false);
    setIsAnimating(false);
    setProgress(0);
    setStatusText("Awaiting input");
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  return (
    <div className="h-screen w-screen bg-background text-foreground flex flex-col overflow-hidden relative">

      {/* ===== FLOATING SIDEBAR ===== */}
      {/* Toggle button — always visible */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="absolute top-4 left-4 z-40 w-9 h-9 rounded-md border bg-card text-card-foreground shadow-sm flex items-center justify-center hover:bg-accent hover:text-accent-foreground transition-colors"
      >
        {sidebarOpen ? <CloseIcon /> : <MenuIcon />}
      </button>

      {/* Sidebar panel */}
      <div
        className={`absolute top-0 left-0 h-full w-72 bg-card border-r border-border shadow-2xl z-30 flex flex-col transition-transform duration-300 ease-in-out ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="p-4 pt-16 flex-1 flex flex-col gap-4 overflow-hidden">
          <h2 className="text-lg font-bold tracking-tight">Obamify Studio</h2>

          {/* New Image */}
          <button
            onClick={() => { handleNewImage(); fileInputRef.current?.click(); }}
            className="w-full inline-flex items-center justify-center gap-2 h-9 px-4 py-2 rounded-md text-sm font-medium bg-primary text-primary-foreground shadow hover:bg-primary/90 transition-colors"
          >
            <PlusIcon className="w-4 h-4" />
            New Morph
          </button>

          {/* Theme toggle */}
          <button
            onClick={() => setIsDark(!isDark)}
            className="w-full inline-flex items-center justify-center gap-2 h-9 px-4 py-2 rounded-md text-sm font-medium border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground transition-colors"
          >
            {isDark ? <SunIcon className="w-4 h-4" /> : <MoonIcon className="w-4 h-4" />}
            {isDark ? "Light Mode" : "Dark Mode"}
          </button>

          {/* History */}
          <div className="flex-1 overflow-hidden flex flex-col">
            <h3 className="text-sm font-semibold text-muted-foreground mb-2">History</h3>
            <div className="flex-1 overflow-y-auto space-y-2 pr-1">
              {history.length === 0 && (
                <p className="text-xs text-muted-foreground/60 italic">No morphs yet</p>
              )}
              {history.map((item) => (
                <div key={item.id} className="flex items-center gap-3 p-2 rounded-lg bg-muted/30 border border-border/50">
                  <img
                    src={item.thumbnail}
                    alt="morph result"
                    className="w-10 h-10 rounded object-cover flex-shrink-0"
                    style={{ imageRendering: "pixelated" }}
                  />
                  <div className="min-w-0">
                    <p className="text-xs font-medium truncate">Morph #{history.length - history.indexOf(item)}</p>
                    <p className="text-[10px] text-muted-foreground">{item.timestamp}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Backdrop when sidebar is open */}
      {sidebarOpen && (
        <div
          className="absolute inset-0 bg-black/30 z-20"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ===== STATUS OVERLAY — top right ===== */}
      {hasStarted && (
        <div className="absolute top-4 right-6 z-10 flex items-center gap-3">
          {isAnimating && (
            <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          )}
          <span className="text-xs font-mono text-muted-foreground">{statusText}</span>
        </div>
      )}

      {/* ===== MAIN CANVAS ===== */}
      <div className="flex-1 flex items-center justify-center px-4 pt-4 pb-2">
        <div className="relative w-full h-full max-w-[min(78vh,80vw)] max-h-[min(78vh,80vw)] aspect-square rounded-2xl overflow-hidden bg-black/40 border border-border/30 shadow-2xl flex items-center justify-center">
          {!hasStarted && (
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10 gap-4">
              <div className="w-16 h-16 rounded-full border-2 border-dashed border-white/20 flex items-center justify-center">
                <UploadIcon className="w-6 h-6 text-white/30" />
              </div>
              <span className="text-white/30 font-mono text-sm tracking-widest uppercase">
                Upload an image to begin
              </span>
            </div>
          )}
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ imageRendering: "pixelated" }}
          ></canvas>
        </div>
      </div>

      {/* ===== UPLOAD BUTTON — below canvas, not overlapping ===== */}
      <div className="flex justify-center pb-5">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleUpload}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isAnimating}
          className="inline-flex items-center justify-center gap-2 h-9 px-4 py-2 rounded-md text-sm font-medium bg-primary text-primary-foreground shadow hover:bg-primary/90 transition-colors disabled:pointer-events-none disabled:opacity-50"
        >
          <UploadIcon className="w-4 h-4" />
          {isAnimating ? "Morphing..." : hasStarted ? "Upload New Image" : "Upload Image"}
        </button>
      </div>

    </div>
  );
}
