"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import obamaImg from "./ref-img/modih.jpeg";

const REFERENCE_IMAGE_PATH = obamaImg.src;
const SIDELEN = 256;
const SIM_STEPS_PER_FRAME = 2;
const MAX_FRAMES = 900;

const workerCode = `
const SIDELEN = ${SIDELEN};
const PERSONAL_SPACE = 0.95;
const MAX_VELOCITY = 6.0;
const ALIGNMENT_FACTOR = 0.8;
const DAMPING = 0.97;
const DST_FORCE = 0.13;

function factorCurve(x) {
  return Math.min(x * x * x, 1000.0);
}

function colorSortKey(r, g, b) {
  const lum = r * 0.299 + g * 0.587 + b * 0.114;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const chroma = max - min;
  let hue = 0;
  if (chroma > 5) {
    if (max === r) hue = ((g - b) / chroma + 6) % 6;
    else if (max === g) hue = (b - r) / chroma + 2;
    else hue = (r - g) / chroma + 4;
    hue /= 6;
  }
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

class MorphSim {
  constructor(n, sidelen, assignments) {
    this.n = n;
    this.sidelen = sidelen;
    this.gridSize = Math.sqrt(n);
    this.pixelSize = sidelen / this.gridSize;

    // Use interleaved X,Y arrays for easy WebGL transfer
    this.pos = new Float32Array(n * 2);
    this.dst = new Float32Array(n * 2);
    this.vel = new Float32Array(n * 2);
    this.acc = new Float32Array(n * 2);
    this.age = new Uint32Array(n);

    const ps = this.pixelSize;
    const gs = this.gridSize;

    for (let i = 0; i < n; i++) {
      const x = (i % gs + 0.5) * ps;
      const y = (((i / gs) | 0) + 0.5) * ps;
      this.pos[i*2] = x;
      this.pos[i*2+1] = y;
    }

    for (let dstIdx = 0; dstIdx < n; dstIdx++) {
      const srcIdx = assignments[dstIdx];
      const dx = (dstIdx % gs + 0.5) * ps;
      const dy = (((dstIdx / gs) | 0) + 0.5) * ps;
      this.dst[srcIdx*2] = dx;
      this.dst[srcIdx*2+1] = dy;
    }
  }

  step() {
    const n = this.n;
    const gs = this.gridSize;
    const ps = this.pixelSize;
    const sl = this.sidelen;
    const personalSpace = ps * PERSONAL_SPACE;
    const halfPS = personalSpace * 0.5;

    const gridLen = gs * gs;
    const gridCells = new Array(gridLen);
    for (let i = 0; i < gridLen; i++) gridCells[i] = [];

    for (let i = 0; i < n; i++) {
      const px = this.pos[i*2], py = this.pos[i*2+1];
      const gx = Math.max(0, Math.min(gs - 1, Math.floor(px / ps)));
      const gy = Math.max(0, Math.min(gs - 1, Math.floor(py / ps)));
      gridCells[(gy * gs + gx) | 0].push(i);
    }

    for (let i = 0; i < n; i++) {
      this.acc[i*2] = 0;
      this.acc[i*2+1] = 0;
    }

    for (let i = 0; i < n; i++) {
      const elapsed = this.age[i] / 60.0;
      const factor = factorCurve(elapsed * DST_FORCE);
      const dx = this.dst[i*2] - this.pos[i*2];
      const dy = this.dst[i*2+1] - this.pos[i*2+1];
      const dist = Math.sqrt(dx * dx + dy * dy);
      this.acc[i*2] += (dx * dist * factor) / sl;
      this.acc[i*2+1] += (dy * dist * factor) / sl;
    }

    for (let i = 0; i < n; i++) {
      const px = this.pos[i*2], py = this.pos[i*2+1];
      if (px < halfPS) this.acc[i*2] += (halfPS - px) / halfPS;
      else if (px > sl - halfPS) this.acc[i*2] -= (px - (sl - halfPS)) / halfPS;
      if (py < halfPS) this.acc[i*2+1] += (halfPS - py) / halfPS;
      else if (py > sl - halfPS) this.acc[i*2+1] -= (py - (sl - halfPS)) / halfPS;
    }

    for (let i = 0; i < n; i++) {
      const px = this.pos[i*2], py = this.pos[i*2+1];
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
            const ox = this.pos[j*2] - px;
            const oy = this.pos[j*2+1] - py;
            const d = Math.sqrt(ox * ox + oy * oy);

            let weight = 0;
            if (d > 0.001 && d < personalSpace) {
              weight = (1.0 / d) * (personalSpace - d) / personalSpace;
              this.acc[i*2] -= ox * weight;
              this.acc[i*2+1] -= oy * weight;
            } else if (d <= 0.001) {
              this.acc[i*2] += (Math.random() - 0.5) * 0.1;
              this.acc[i*2+1] += (Math.random() - 0.5) * 0.1;
              weight = 0.5;
            }

            if (weight > 0) {
              avgVx += this.vel[j*2] * weight;
              avgVy += this.vel[j*2+1] * weight;
              count += weight;
            }
          }
        }
      }

      if (count > 0) {
        avgVx /= count;
        avgVy /= count;
        this.acc[i*2] += (avgVx - this.vel[i*2]) * ALIGNMENT_FACTOR;
        this.acc[i*2+1] += (avgVy - this.vel[i*2+1]) * ALIGNMENT_FACTOR;
      }
    }

    for (let i = 0; i < n; i++) {
      this.vel[i*2] = (this.vel[i*2] + this.acc[i*2]) * DAMPING;
      this.vel[i*2+1] = (this.vel[i*2+1] + this.acc[i*2+1]) * DAMPING;
      this.pos[i*2] += Math.max(-MAX_VELOCITY, Math.min(MAX_VELOCITY, this.vel[i*2]));
      this.pos[i*2+1] += Math.max(-MAX_VELOCITY, Math.min(MAX_VELOCITY, this.vel[i*2+1]));
      this.age[i]++;
    }
  }

  isSettled() {
    let settled = 0;
    const thresh = this.pixelSize * 0.5;
    const distSq = thresh * thresh;
    for (let i = 0; i < this.n; i++) {
      const dx = this.dst[i*2] - this.pos[i*2];
      const dy = this.dst[i*2+1] - this.pos[i*2+1];
      if (dx * dx + dy * dy < distSq) settled++;
    }
    return settled > this.n * 0.95;
  }
}

let sim = null;
let loopActive = false;
let frameCount = 0;
let finalDst = null;

self.onmessage = function(e) {
  const data = e.data;
  if (data.type === "init") {
    self.postMessage({ type: "status", status: "Sorting pixels by color..." });
    const assignments = computeAssignments(data.sourcePixels, data.targetPixels, SIDELEN);
    
    self.postMessage({ type: "status", status: "Initializing boids physics..." });
    sim = new MorphSim(SIDELEN * SIDELEN, SIDELEN, assignments);
    frameCount = 0;
    finalDst = sim.dst; // cache ideal placements
    
    self.postMessage({ type: "ready", assignments: assignments });
  } 
  else if (data.type === "start") {
    loopActive = true;
    (function tick() {
      if (!loopActive) return;
      for (let i = 0; i < ${SIM_STEPS_PER_FRAME}; i++) sim.step();
      frameCount += ${SIM_STEPS_PER_FRAME};
      
      const posCopy = new Float32Array(sim.pos); // Avoid transfer detachment of internal array
      
      let isSettled = false;
      if (frameCount > 200 && frameCount % 20 === 0) isSettled = sim.isSettled();
      if (frameCount >= ${MAX_FRAMES} || isSettled) {
         // Final clean frame
         loopActive = false;
         self.postMessage({ type: "complete", positions: finalDst }, [finalDst.buffer]); 
         return;
      }
      
      self.postMessage({ type: "frame", positions: posCopy, progress: Math.min(100, (frameCount / ${MAX_FRAMES}) * 100) }, [posCopy.buffer]);
      setTimeout(tick, 0); // Recursively dispatch to yield event loop
    })();
  }
  else if (data.type === "stop") {
    loopActive = false;
  }
};
`;

// ============================================================================
// WEBGL RENDERER (Hardware Graphics Acceleration)
// ============================================================================

const vsShaderSrc = `#version 300 es
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
uniform float uSidelen;
void main() {
    vColor = aColor;
    float x = (aPosition.x / uSidelen) * 2.0 - 1.0;
    float y = 1.0 - (aPosition.y / uSidelen) * 2.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
    gl_PointSize = 1.5; // Natural sub-pixel coverage without CPU passes
}`;

const fsShaderSrc = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(vColor, 1.0);
}`;

function initWebGL(gl) {
  const compileShader = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vsShaderSrc));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fsShaderSrc));
  gl.linkProgram(program);
  gl.useProgram(program);

  const posBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  const colBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuffer);
  gl.enableVertexAttribArray(1);
  // RGB Unsigned bytes normalized
  gl.vertexAttribPointer(1, 3, gl.UNSIGNED_BYTE, true, 0, 0);

  gl.uniform1f(gl.getUniformLocation(program, "uSidelen"), SIDELEN);

  return { posBuffer, colBuffer };
}

// ============================================================================
// ICONS
// ============================================================================

const UploadIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>);
const DownloadIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>);
const MenuIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="4" y1="6" x2="20" y2="6"></line><line x1="4" y1="12" x2="20" y2="12"></line><line x1="4" y1="18" x2="20" y2="18"></line></svg>);
const CloseIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>);
const SunIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>);
const MoonIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>);
const PlusIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>);

// ============================================================================
// MAIN APPLICATION DASHBOARD
// ============================================================================

export default function MorphDashboard() {
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const workerRef = useRef(null);
  const webglStateRef = useRef(null);

  const [isAnimating, setIsAnimating] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [statusText, setStatusText] = useState("Awaiting input");
  const [progress, setProgress] = useState(0);
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isDark, setIsDark] = useState(true);
  const [history, setHistory] = useState([]);
  const [gifBlobUrl, setGifBlobUrl] = useState(null);
  const [isGeneratingGif, setIsGeneratingGif] = useState(false);
  const gifRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  useEffect(() => {
    document.documentElement.classList.add("dark");
    
    // Initialize Web Worker
    const blob = new Blob([workerCode], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    workerRef.current = new Worker(url);

    return () => {
      workerRef.current?.terminate();
      URL.revokeObjectURL(url);
    };
  }, []);

  const loadImgPixels = async (src) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = src;
    });
    
    const off = document.createElement('canvas');
    off.width = SIDELEN; off.height = SIDELEN;
    const ctx = off.getContext('2d');
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, SIDELEN, SIDELEN);
    
    const scale = Math.max(SIDELEN / img.width, SIDELEN / img.height);
    const w = img.width * scale, h = img.height * scale;
    ctx.drawImage(img, (SIDELEN - w) / 2, (SIDELEN - h) / 2, w, h);
    
    // Only return RGB for performance
    const idata = ctx.getImageData(0, 0, SIDELEN, SIDELEN).data;
    const rgb = new Uint8Array(SIDELEN * SIDELEN * 3);
    for (let i = 0; i < SIDELEN * SIDELEN; i++) {
      rgb[i * 3] = idata[i * 4];
      rgb[i * 3 + 1] = idata[i * 4 + 1];
      rgb[i * 3 + 2] = idata[i * 4 + 2];
    }
    return rgb;
  };

  const handleUpload = async (e) => {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];
    const sourceUrl = URL.createObjectURL(file);
    e.target.value = ""; // reset
    
    // UI Resets
    setHasStarted(true);
    setIsAnimating(true);
    setProgress(0);
    setStatusText("Loading images...");
    setGifBlobUrl(null);
    if (gifRef.current) gifRef.current.abort();

    try {
      const sourcePixels = await loadImgPixels(sourceUrl);
      const targetPixels = await loadImgPixels(REFERENCE_IMAGE_PATH);
      
      const gl = canvasRef.current?.getContext('webgl2', { preserveDrawingBuffer: true }); // Need preserve for GIF recording
      canvasRef.current.width = SIDELEN;
      canvasRef.current.height = SIDELEN;

      if (!webglStateRef.current && gl) {
        webglStateRef.current = initWebGL(gl);
      }
      
      const { colBuffer, posBuffer } = webglStateRef.current;
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sourcePixels, gl.STATIC_DRAW);

        // Instantiate GIF recorder
        const workerRequest = await fetch('https://cdn.jsdelivr.net/npm/gif.js/dist/gif.worker.js');
        const workerText = await workerRequest.text();
        const workerBlob = new Blob([workerText], { type: 'application/javascript' });
        const gif = new window.GIF({
          workers: 4,
          quality: 10,
          width: SIDELEN,
          height: SIDELEN,
          workerScript: URL.createObjectURL(workerBlob)
        });
        
        gif.on('finished', (blob) => {
          setGifBlobUrl(URL.createObjectURL(blob));
          setStatusText("GIF Rendering Complete!");
          setIsGeneratingGif(false);
        });

        workerRef.current.onmessage = (e) => {
          const msg = e.data;
          if (msg.type === "status") {
            setStatusText(msg.status);
          } else if (msg.type === "ready") {
            setStatusText("Magic is on it's way...");
            workerRef.current.postMessage({ type: "start" });
          } else if (msg.type === "frame" || msg.type === "complete") {
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clearColor(0,0,0,1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, msg.positions, gl.DYNAMIC_DRAW);
            gl.drawArrays(gl.POINTS, 0, SIDELEN * SIDELEN);
            
            // Capture frame roughly every 2nd or 3rd frame (so GIF isn't horribly bloated)
            // msg.progress maps 0 to 100 over 900 frames
            if (msg.type === "frame") {
               setProgress(msg.progress);
               // Add frame
               gif.addFrame(canvasRef.current, { copy: true, delay: 33 });
            } else {
               setProgress(100);
               setIsAnimating(false);
               setStatusText("Modihfication completed");
               gif.addFrame(canvasRef.current, { copy: true, delay: 2000 }); // Hold last frame
               gifRef.current = gif;
               try {
                  const thumb = canvasRef.current.toDataURL("image/jpeg", 0.6);
                  setHistory(prev => [{ id: Date.now(), thumbnail: thumb, timestamp: new Date().toLocaleTimeString() }, ...prev]);
               } catch(e) {}
            }
          }
        };

        workerRef.current.postMessage({ type: "init", sourcePixels, targetPixels });
      } catch (err) {
        setStatusText("Error: " + err.message);
        setIsAnimating(false);
      }
    };

    const handleNewImage = () => {
      workerRef.current?.postMessage({ type: "stop" });
      setHasStarted(false);
      setIsAnimating(false);
      setProgress(0);
      setStatusText("Awaiting input");
      setGifBlobUrl(null);
      if (gifRef.current && typeof gifRef.current.abort === 'function') gifRef.current.abort();
      
      const gl = canvasRef.current?.getContext('webgl2');
      if (gl) {
        gl.clearColor(0,0,0,1);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
    };

    const handleDownloadGif = async () => {
      if (gifBlobUrl) {
         // Already generated, download it
         const a = document.createElement("a");
         a.href = gifBlobUrl;
         a.download = "modih-morph.gif";
         a.click();
         return;
      }

      // Need to render
      if (gifRef.current && !isGeneratingGif) {
         setIsGeneratingGif(true);
         setStatusText("Encoding GIF (this may take a few seconds)...");
         gifRef.current.render();
      }
    };

    return (
      <div className="h-screen w-screen bg-background text-foreground flex flex-col overflow-hidden relative">

      {/* ===== FLOATING SIDEBAR ===== */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="absolute top-4 left-4 z-40 w-9 h-9 rounded-md border bg-card text-card-foreground shadow-sm flex items-center justify-center hover:bg-accent hover:text-accent-foreground transition-colors"
      >
        {sidebarOpen ? <CloseIcon /> : <MenuIcon />}
      </button>

      <div
        className={`absolute top-0 left-0 h-full w-72 bg-card border-r border-border shadow-2xl z-30 flex flex-col transition-transform duration-300 ease-in-out ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="p-4 pt-16 flex-1 flex flex-col gap-4 overflow-hidden">
          <h2 className="text-lg font-bold tracking-tight">Modihfy</h2>

          <button
            onClick={() => { handleNewImage(); fileInputRef.current?.click(); }}
            className="w-full inline-flex items-center justify-center gap-2 h-9 px-4 py-2 rounded-md text-sm font-medium bg-primary text-primary-foreground shadow hover:bg-primary/90 transition-colors"
          >
            <PlusIcon className="w-4 h-4" />
            New Morph
          </button>

          <button
            onClick={() => setIsDark(!isDark)}
            className="w-full inline-flex items-center justify-center gap-2 h-9 px-4 py-2 rounded-md text-sm font-medium border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground transition-colors"
          >
            {isDark ? <SunIcon className="w-4 h-4" /> : <MoonIcon className="w-4 h-4" />}
            {isDark ? "Light Mode" : "Dark Mode"}
          </button>

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

      {sidebarOpen && (
        <div
          className="absolute inset-0 bg-black/30 z-20"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ===== HEADER / STATUS ===== */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
        <h1 className="text-xl font-bold tracking-tight text-foreground/80 opacity-50">Only for true chuds</h1>
      </div>

      {hasStarted && (
        <div className="absolute top-4 right-6 z-10 flex items-center gap-3">
          {isAnimating && (
            <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          )}
          <span className="text-xs font-mono text-muted-foreground">{statusText}</span>
        </div>
      )}

      {/* ===== DEFAULT LAYOUT CANVAS ===== */}
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

      {/* ===== ACTIONS BAR ===== */}
      <div className="flex justify-center pb-8 gap-4 px-6 relative z-10">
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
          className="inline-flex items-center justify-center gap-2 h-10 px-6 rounded-md text-sm font-medium bg-primary text-primary-foreground shadow-lg hover:bg-primary/90 transition-all disabled:pointer-events-none disabled:opacity-50 hover:-translate-y-0.5 active:translate-y-0"
        >
          <UploadIcon className="w-4 h-4" />
          {isAnimating ? "Modihfying. Please wait..." : hasStarted ? "Upload New Source Image" : "Upload Image"}
        </button>

        <button
          onClick={handleDownloadGif}
          disabled={!hasStarted || isGeneratingGif || isAnimating}
          className="inline-flex items-center justify-center gap-2 h-10 px-6 rounded-md text-sm font-medium bg-accent text-accent-foreground shadow-lg hover:bg-accent/90 transition-all disabled:opacity-50 hover:-translate-y-0.5 active:translate-y-0"
        >
          <DownloadIcon className="w-4 h-4" />
          {isGeneratingGif ? "Generating..." : isAnimating ? "Hold on tight..." : gifBlobUrl ? "Save GIF" : "Download GIF"}
        </button>
      </div>

    </div>
  );
}
