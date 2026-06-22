import { FaceLandmarker, HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.19/vision_bundle.mjs";

const video = document.getElementById('webcam');
const stage = document.querySelector('.camera-stage');
const overlay = document.getElementById('overlayCanvas');
const octx = overlay.getContext('2d');
const handCursor = document.getElementById('handCursor');
const filterToast = document.getElementById('filterToast');
const gallery = document.getElementById('galleryContainer');
const flash = document.getElementById('flash-overlay');
const statusDot = document.getElementById('statusDot');
const gestureLabel = document.getElementById('gestureLabel');
const centerCountdown = document.getElementById('centerCountdown');
const micIcon = document.getElementById('micIcon');
const loader = document.getElementById('loader');
const vfCorners = document.querySelectorAll('.vf-corner');
const reticle = document.getElementById('reticle');
const spectrumEl = document.getElementById('spectrum');
const voiceCmd = document.getElementById('voiceCmd');

// Telemetry refs
const telFace = document.getElementById('telFace');
const telHand = document.getElementById('telHand');
const telNodBar = document.getElementById('telNodBar');
const telNodVal = document.getElementById('telNodVal');
const telZoom = document.getElementById('telZoom');
const telFps = document.getElementById('telFps');

let faceLandmarker = null;
let handLandmarker = null;
let lastVideoTime = -1;
let isShutterCooldown = false;
let currentZoom = 1.0;
let targetZoom = 1.0;

let cameraState = 'STANDBY';
let armCountdownStart = 0;
let snapCountdownStart = 0;
let lastCountNumber = -1;

const ARM_SECONDS = 3; // hold-to-arm
const SNAP_SECONDS = 5; // photo countdown

// Feature flags (driven by settings)
const FX = { telemetry: true, voice: true, reticle: true, viewfinder: true, effects: true, glasses: false, cap: false, cursor: false };

// Colour filters cycled by swipe
const FILTERS = [
  { name: 'NONE', css: 'none' },
  { name: 'MONO', css: 'grayscale(1) contrast(1.05)' },
  { name: 'NOIR', css: 'grayscale(1) contrast(1.45) brightness(0.92)' },
  { name: 'SEPIA', css: 'sepia(0.7) contrast(1.05) brightness(1.02)' },
  { name: 'WARM', css: 'saturate(1.35) sepia(0.18) brightness(1.04)' },
  { name: 'COOL', css: 'saturate(1.2) hue-rotate(15deg) brightness(1.02)' },
  { name: 'VIVID', css: 'saturate(1.7) contrast(1.12)' },
  { name: 'CYBER', css: 'invert(1) hue-rotate(180deg) saturate(1.3)' }
];
let filterIndex = 0;

function applyFilter() {
  video.style.filter = FILTERS[filterIndex].css;
  filterToast.textContent = FILTERS[filterIndex].name;
  filterToast.classList.add('show');
  clearTimeout(applyFilter._t);
  applyFilter._t = setTimeout(() => filterToast.classList.remove('show'), 1100);
}

function cycleFilter(dir) {
  filterIndex = (filterIndex + dir + FILTERS.length) % FILTERS.length;
  applyFilter();
}

// Live telemetry signals
let faceTracked = false;
let handTracked = false;
let pinchSignal = 0; // 0..1 how closed the pinch is
let fpsValue = 0;
let lastFrameStamp = 0;
let lastFace = null; // cached face landmarks for AR + snapshot

// ---------- 1. PINCH + GESTURE DETECTION ----------
function getDistance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

// Pinch ratio = thumb-tip<->index-tip gap normalized by palm length
// (wrist <-> middle MCP). Distance-invariant: same value near or far.
const PINCH_ON = 0.60; // ratio below this = pinched (more sensitive)
const PINCH_OFF = 0.82; // ratio above this = released (hysteresis)
let wasPinched = false;

function getPinchRatio(landmarks) {
  const lm = landmarks[0];
  const gap = getDistance(lm[4], lm[8]); // thumb tip -> index tip
  const palm = getDistance(lm[0], lm[9]) || 0.0001; // wrist -> middle MCP
  return gap / palm;
}

function detectPinch(landmarks) {
  const ratio = getPinchRatio(landmarks);
  pinchSignal = Math.max(0, Math.min(1, (PINCH_OFF - ratio) / (PINCH_OFF - PINCH_ON)));
  let fired = false;
  if (!wasPinched && ratio < PINCH_ON) { wasPinched = true;
    fired = true; }
  else if (wasPinched && ratio > PINCH_OFF) { wasPinched = false; }
  return fired;
}

function resetPinch() { wasPinched = false;
  pinchSignal = 0; }

// Pose classification (one of POINT / TWO / THREE / OPEN / OTHER)
function getPose(landmarks) {
  const lm = landmarks[0];
  const up = (tip, pip) => lm[tip].y < lm[pip].y - 0.01; // small margin = snappier
  const i = up(8, 6),
    m = up(12, 10),
    r = up(16, 14),
    p = up(20, 18);
  if (i && !m && !r && !p) return 'POINT';
  if (i && m && !r && !p) return 'TWO';
  if (i && m && r && !p) return 'THREE';
  if (i && m && r && p) return 'OPEN';
  return 'OTHER';
}

// STANDBY-only zoom from thumb/index spread (distance-invariant).
function handlePinchZoom(landmarks) {
  const ratio = getPinchRatio(landmarks);
  targetZoom = Math.min(3.0, Math.max(1.0, 1.0 + (ratio - 0.4) * 2.2));
}

// ---------- SWIPE -> colour filters (open palm, fast horizontal move) ----------
let swipeRef = null; // { x, t }
let lastSwipe = 0;
const SWIPE_DIST = 0.16; // fraction of frame width
const SWIPE_TIME = 450; // ms window
function handleSwipe(landmarks) {
  const lm = landmarks[0];
  const x = (lm[0].x + lm[9].x) / 2; // palm centre x (source space)
  const now = Date.now();
  if (!swipeRef || now - swipeRef.t > SWIPE_TIME) { swipeRef = { x, t: now }; return; }
  const dx = x - swipeRef.x;
  if (Math.abs(dx) > SWIPE_DIST && now - lastSwipe > 700) {
    lastSwipe = now;
    // source x is mirrored on screen: source-right (dx>0) shows as screen-left
    cycleFilter(dx > 0 ? -1 : 1);
    swipeRef = { x, t: now };
  }
}

function resetSwipe() { swipeRef = null; }

// ---------- HAND CURSOR (point to move, push to click) ----------
// Map a source-normalized point to viewport coords matching the displayed
// (cover-cropped, mirrored, zoomed) video so the cursor sits on the fingertip.
function toScreen(nx, ny) {
  const rect = stage.getBoundingClientRect();
  const W = rect.width,
    H = rect.height;
  const vw = video.videoWidth || W,
    vh = video.videoHeight || H;
  const s = Math.max(W / vw, H / vh);
  const dw = vw * s,
    dh = vh * s;
  const ox = (W - dw) / 2,
    oy = (H - dh) / 2;
  let x = ox + nx * dw;
  let y = oy + ny * dh;
  x = W - x; // mirror
  x = W / 2 + (x - W / 2) * currentZoom; // zoom about centre
  y = H / 2 + (y - H / 2) * currentZoom;
  return { x: rect.left + x, y: rect.top + y };
}

let cursorActive = false;
let pushBaseline = null,
  wasClicking = false;
const PUSH_TRIGGER = 0.045; // forward depth delta to register a click
const PUSH_RELEASE = 0.020;

function showCursor(landmarks) {
  const lm = landmarks[0];
  const tip = lm[8];
  const pos = toScreen(tip.x, tip.y);
  handCursor.style.left = pos.x + 'px';
  handCursor.style.top = pos.y + 'px';
  handCursor.classList.add('show');
  cursorActive = true;
  
  // Push detection via relative depth (index tip vs wrist). Forward = more negative z.
  const relZ = lm[8].z - lm[0].z;
  if (pushBaseline === null) pushBaseline = relZ;
  const push = pushBaseline - relZ; // >0 when pushing toward camera
  const fill = handCursor.querySelector('.hc-fill');
  const pp = Math.max(0, Math.min(1, push / PUSH_TRIGGER));
  fill.style.width = fill.style.height = (pp * 40) + 'px';
  
  if (!wasClicking && push > PUSH_TRIGGER) {
    wasClicking = true;
    handCursor.classList.add('click');
    clickAt(pos.x, pos.y);
  } else if (wasClicking && push < PUSH_RELEASE) {
    wasClicking = false;
    handCursor.classList.remove('click');
  }
  // drift baseline slowly while not pushing so it self-calibrates
  if (push < PUSH_RELEASE) pushBaseline = pushBaseline * 0.9 + relZ * 0.1;
}

function hideCursor() {
  if (!cursorActive) return;
  handCursor.classList.remove('show', 'click');
  cursorActive = false;
  pushBaseline = null;
  wasClicking = false;
}

function clickAt(x, y) {
  const el = document.elementFromPoint(x, y);
  if (el) el.click();
}

// ---------- AR ACCESSORIES (drawn on the cover-aligned overlay canvas) ----------
function syncOverlay() {
  const vw = video.videoWidth,
    vh = video.videoHeight;
  if (vw && (overlay.width !== vw || overlay.height !== vh)) { overlay.width = vw;
    overlay.height = vh; }
}

function drawAR(face) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!face) return;
  const vw = overlay.width,
    vh = overlay.height;
  if (FX.cap) drawCap(octx, face, vw, vh);
  if (FX.glasses) drawGlasses(octx, face, vw, vh);
}

function drawGlasses(ctx, f, vw, vh) {
  const L = f[33],
    R = f[263]; // outer eye corners
  const lx = L.x * vw,
    ly = L.y * vh,
    rx = R.x * vw,
    ry = R.y * vh;
  const cx = (lx + rx) / 2,
    cy = (ly + ry) / 2;
  const eye = Math.hypot(rx - lx, ry - ly);
  const ang = Math.atan2(ry - ly, rx - lx);
  const off = eye * 0.52,
    lensR = eye * 0.34;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(ang);
  ctx.lineWidth = Math.max(3, eye * 0.05);
  ctx.strokeStyle = 'rgba(12,14,20,0.95)';
  ctx.fillStyle = 'rgba(18,22,30,0.5)';
  for (const sgn of [-1, 1]) {
    ctx.beginPath();
    ctx.arc(sgn * off, 0, lensR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(-off + lensR, 0);
  ctx.lineTo(off - lensR, 0);
  ctx.stroke(); // bridge
  ctx.beginPath();
  ctx.moveTo(-off - lensR, -eye * 0.05);
  ctx.lineTo(-off - lensR - eye * 0.55, -eye * 0.12);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(off + lensR, -eye * 0.05);
  ctx.lineTo(off + lensR + eye * 0.55, -eye * 0.12);
  ctx.stroke();
  ctx.restore();
}

function drawCap(ctx, f, vw, vh) {
  const Lp = f[234],
    Rp = f[454],
    top = f[10];
  const lx = Lp.x * vw,
    ly = Lp.y * vh,
    rx = Rp.x * vw,
    ry = Rp.y * vh;
  const faceW = Math.hypot(rx - lx, ry - ly);
  const ang = Math.atan2(ry - ly, rx - lx);
  const tx = top.x * vw,
    ty = top.y * vh;
  ctx.save();
  ctx.translate(tx, ty);
  ctx.rotate(ang);
  ctx.fillStyle = 'rgba(22,34,68,0.92)';
  ctx.beginPath();
  ctx.ellipse(0, -faceW * 0.16, faceW * 0.56, faceW * 0.46, 0, Math.PI, 0);
  ctx.fill(); // dome
  ctx.beginPath();
  ctx.ellipse(0, -faceW * 0.02, faceW * 0.62, faceW * 0.12, 0, 0, Math.PI * 2);
  ctx.fill(); // brim
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.beginPath();
  ctx.arc(0, -faceW * 0.30, faceW * 0.05, 0, Math.PI * 2);
  ctx.fill(); // button
  ctx.restore();
}

// ---------- 3. CAMERA CONTROLS ----------
function startSnapCountdown() {
  if (cameraState === 'SNAP_COUNTDOWN' || isShutterCooldown) return;
  cameraState = 'SNAP_COUNTDOWN';
  snapCountdownStart = Date.now();
  lastCountNumber = -1;
}

function executeSnap() {
  if (isShutterCooldown) return;
  isShutterCooldown = true;
  flash.classList.add('active');
  setTimeout(() => flash.classList.remove('active'), 50);
  takeSnapshot();
  setTimeout(() => isShutterCooldown = false, 1500);
}

function takeSnapshot() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const cx = c.getContext('2d');
  cx.translate(w, 0);
  cx.scale(-1, 1);
  cx.filter = FILTERS[filterIndex].css; // bake colour filter
  cx.drawImage(video, 0, 0, w, h);
  cx.filter = 'none';
  if (FX.glasses || FX.cap) cx.drawImage(overlay, 0, 0, w, h); // bake AR
  
  const dataUrl = c.toDataURL('image/png');
  const wrapper = document.createElement('div');
  wrapper.className = 'snapshot-item';
  
  const img = document.createElement('img');
  img.src = dataUrl;
  const dlOverlay = document.createElement('div');
  dlOverlay.className = 'download-overlay';
  dlOverlay.innerHTML = `<svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>`;
  
  wrapper.appendChild(img);
  wrapper.appendChild(dlOverlay);
  wrapper.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `Aperture_${Date.now()}.png`;
    a.click();
  });
  
  gallery.prepend(wrapper);
  if (gallery.children.length > 20) gallery.removeChild(gallery.lastChild);
}

// ---------- 4. UI RENDERER ----------
function updateUI() {
  let label = '';
  let colorClass = '';
  let cornerColor = 'rgba(255, 255, 255, 0.4)';
  centerCountdown.style.display = 'none';
  
  if (isShutterCooldown) {
    label = 'SHOT';
    colorClass = 'alert';
    cornerColor = 'var(--red)';
  } else if (cameraState === 'SNAP_COUNTDOWN') {
    const left = Math.ceil(SNAP_SECONDS - (Date.now() - snapCountdownStart) / 1000);
    label = 'SHOOT';
    colorClass = 'alert';
    cornerColor = 'var(--red)';
    
    centerCountdown.style.display = 'flex';
    centerCountdown.textContent = left;
    if (left !== lastCountNumber) {
      centerCountdown.style.animation = 'none';
      void centerCountdown.offsetWidth;
      centerCountdown.style.animation = 'popIn 1s infinite cubic-bezier(0.2, 0, 0, 1)';
      lastCountNumber = left;
    }
  } else if (cameraState === 'ARM_COUNTDOWN') {
    const left = Math.ceil(ARM_SECONDS - (Date.now() - armCountdownStart) / 1000);
    label = `ARM ${left}`;
    colorClass = 'warning';
    cornerColor = 'var(--amber)';
  } else if (cameraState === 'ARMED') {
    label = 'PINCH';
    colorClass = 'active';
    cornerColor = 'var(--green)';
  } else {
    label = 'STANDBY';
    colorClass = 'standby';
    cornerColor = 'rgba(255, 255, 255, 0.1)';
  }
  
  gestureLabel.textContent = label;
  statusDot.className = `status-dot ${colorClass}`;
  vfCorners.forEach(c => c.style.borderColor = cornerColor);
}

// Telemetry HUD updates run every frame for smoothness
function renderTelemetry() {
  reticle.classList.toggle('locked', faceTracked);
  
  telFace.textContent = faceTracked ? 'OK' : '--';
  telFace.className = 'tel-val ' + (faceTracked ? 'on' : 'off');
  telHand.textContent = handTracked ? 'OK' : '--';
  telHand.className = 'tel-val ' + (handTracked ? 'on' : 'off');
  
  const pct = Math.round(pinchSignal * 100);
  telNodBar.style.width = pct + '%';
  telNodBar.style.background = pinchSignal > 0.85 ? 'var(--green)' : pinchSignal > 0.4 ? 'var(--amber)' : 'var(--grey)';
  telNodVal.textContent = String(pct).padStart(2, '0') + '%';
  
  telZoom.textContent = currentZoom.toFixed(2) + 'x';
  telFps.textContent = fpsValue ? String(Math.round(fpsValue)) : '--';
}

// ---------- 5. PREDICTION LOOP ----------
function predictWebcam() {
  currentZoom += (targetZoom - currentZoom) * 0.22; // snappier
  const tf = `scaleX(-1) scale(${currentZoom})`;
  video.style.transform = tf;
  overlay.style.transform = tf;
  
  if (video.currentTime !== lastVideoTime && video.videoWidth > 0 && faceLandmarker && handLandmarker) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    
    // FPS (smoothed)
    if (lastFrameStamp) {
      const inst = 1000 / (now - lastFrameStamp);
      fpsValue = fpsValue ? fpsValue * 0.85 + inst * 0.15 : inst;
    }
    lastFrameStamp = now;
    
    syncOverlay();
    const faceResult = faceLandmarker.detectForVideo(video, now);
    const handResult = handLandmarker.detectForVideo(video, now);
    
    // ----- Hands -----
    handTracked = !!(handResult.landmarks && handResult.landmarks.length > 0);
    let pointing = false;
    
    if (handTracked) {
      const pose = getPose(handResult.landmarks);
      
      // Hand cursor takes priority while pointing
      if (FX.cursor && pose === 'POINT') {
        pointing = true;
        showCursor(handResult.landmarks);
      } else {
        hideCursor();
      }
      
      // Swipe filters with an open palm
      if (pose === 'OPEN') handleSwipe(handResult.landmarks);
      else resetSwipe();
      
      if (!pointing) {
        // Mode switching
        if (pose === 'TWO' && cameraState === 'STANDBY') {
          cameraState = 'ARM_COUNTDOWN';
          armCountdownStart = Date.now();
        } else if (pose === 'THREE') {
          cameraState = 'STANDBY';
          resetPinch();
        }
        
        if (cameraState === 'STANDBY') {
          if (pose === 'OTHER') handlePinchZoom(handResult.landmarks); // zoom only here
          pinchSignal = 0;
        } else if (cameraState === 'ARMED') {
          if (detectPinch(handResult.landmarks)) startSnapCountdown(); // pinch = shoot
        } else {
          pinchSignal = 0;
        }
      } else {
        pinchSignal = 0;
      }
    } else {
      hideCursor();
      resetSwipe();
      resetPinch();
    }
    
    // ----- Timers -----
    if (cameraState === 'ARM_COUNTDOWN' && Date.now() - armCountdownStart >= ARM_SECONDS * 1000) {
      cameraState = 'ARMED';
      resetPinch();
    }
    if (cameraState === 'SNAP_COUNTDOWN' && Date.now() - snapCountdownStart >= SNAP_SECONDS * 1000) {
      executeSnap();
      cameraState = 'ARMED';
      resetPinch();
    }
    
    // ----- Face (framing lock, telemetry, AR) -----
    faceTracked = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);
    lastFace = faceTracked ? faceResult.faceLandmarks[0] : null;
    drawAR((FX.glasses || FX.cap) ? lastFace : null);
    
    updateUI();
  }
  
  renderTelemetry();
  requestAnimationFrame(predictWebcam);
}

// ---------- 6. VOICE COMMANDS (fast / interim) ----------
const COMMANDS = [
  { keys: ['snap', 'cheese', 'shoot', 'capture', 'take it', 'go'], label: 'CAPTURE', run: () => executeSnap() },
  { keys: ['count', 'timer', 'countdown'], label: 'COUNTDOWN', run: () => startSnapCountdown() },
  { keys: ['zoom in', 'closer'], label: 'ZOOM IN', run: () => targetZoom = Math.min(3.0, targetZoom + 0.5) },
  { keys: ['zoom out', 'back out', 'wider'], label: 'ZOOM OUT', run: () => targetZoom = Math.max(1.0, targetZoom - 0.5) },
  { keys: ['zoom full', 'max zoom', 'full zoom'], label: 'MAX ZOOM', run: () => targetZoom = 3.0 },
  { keys: ['reset zoom', 'zoom reset', 'normal'], label: 'RESET ZOOM', run: () => targetZoom = 1.0 }
];

let lastVoiceFire = 0;

function handleVoice(text) {
  const now = Date.now();
  if (now - lastVoiceFire < 600) return false; // debounce repeats from interim+final
  for (const cmd of COMMANDS) {
    if (cmd.keys.some(k => text.includes(k))) {
      cmd.run();
      lastVoiceFire = now;
      voiceCmd.innerHTML = `<b>${cmd.label}</b>`;
      clearTimeout(handleVoice._t);
      handleVoice._t = setTimeout(() => { voiceCmd.textContent = 'READY'; }, 1600);
      return true;
    }
  }
  return false;
}

function initVoiceCommands() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) { voiceCmd.textContent = 'NO MIC'; return; }
  
  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true; // act on partial transcripts -> much faster
  recognition.maxAlternatives = 3; // catch phonetic variants
  recognition.lang = 'en-US';
  
  recognition.onstart = () => { micIcon.classList.add('listening'); };
  recognition.onerror = () => { micIcon.classList.remove('listening'); };
  
  recognition.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const res = event.results[i];
      for (let a = 0; a < res.length; a++) {
        if (handleVoice((res[a].transcript || '').toLowerCase().trim())) return;
      }
    }
  };
  
  recognition.onend = () => {
    micIcon.classList.remove('listening');
    try { recognition.start(); } catch (e) {}
  };
  try { recognition.start(); } catch (e) {}
}

// ---------- 7. AUDIO SPECTRUM METER ----------
const SPECTRUM_BARS = 14;
const sbars = [];
for (let i = 0; i < SPECTRUM_BARS; i++) {
  const b = document.createElement('div');
  b.className = 'sbar';
  spectrumEl.appendChild(b);
  sbars.push(b);
}

let analyser = null,
  audioData = null;

function initAudioMeter(stream) {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    const src = ctx.createMediaStreamSource(stream);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 64;
    analyser.smoothingTimeConstant = 0.7;
    src.connect(analyser);
    audioData = new Uint8Array(analyser.frequencyBinCount);
    if (ctx.state === 'suspended') ctx.resume();
    renderSpectrum();
  } catch (e) {}
}

function renderSpectrum() {
  if (analyser && audioData) {
    analyser.getByteFrequencyData(audioData);
    const step = Math.floor(audioData.length / SPECTRUM_BARS) || 1;
    let peak = 0;
    for (let i = 0; i < SPECTRUM_BARS; i++) {
      const v = audioData[i * step] / 255; // 0..1
      peak = Math.max(peak, v);
      sbars[i].style.height = (6 + v * 94) + '%';
    }
    spectrumEl.classList.toggle('hot', peak > 0.28);
  }
  requestAnimationFrame(renderSpectrum);
}

// ---------- SETTINGS ----------
(function initSettings() {
  const btn = document.getElementById('settingsBtn');
  const panel = document.getElementById('settingsPanel');
  const HUD_KEYS = ['telemetry', 'voice', 'reticle', 'viewfinder', 'effects'];
  
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    panel.classList.toggle('open');
    btn.classList.toggle('open');
  });
  
  document.addEventListener('click', (e) => {
    if (!panel.contains(e.target) && e.target !== btn && !btn.contains(e.target)) {
      panel.classList.remove('open');
      btn.classList.remove('open');
    }
  });
  
  function apply(key, on) {
    FX[key] = on;
    if (HUD_KEYS.includes(key)) document.body.classList.toggle('off-' + key, !on);
    if (key === 'cursor' && !on) hideCursor();
    if ((key === 'glasses' || key === 'cap') && !on && !FX.glasses && !FX.cap) {
      octx.clearRect(0, 0, overlay.width, overlay.height);
    }
  }
  
  panel.querySelectorAll('.toggle-row').forEach((row) => {
    const key = row.dataset.key;
    row.addEventListener('click', () => {
      const on = row.querySelector('.switch').classList.toggle('on');
      apply(key, on);
    });
  });
})();

// ---------- 8. BOOTUP ----------
async function bootML() {
  try {
    const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.19/wasm');
    
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', delegate: 'CPU' },
      outputFaceBlendshapes: false,
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.3,
      minFacePresenceConfidence: 0.3,
      minTrackingConfidence: 0.3
    });
    
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', delegate: 'CPU' },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence: 0.3,
      minTrackingConfidence: 0.3
    });
    
    loader.style.display = 'none';
    requestAnimationFrame(predictWebcam);
    initVoiceCommands();
  } catch (err) {
    document.getElementById('loaderText').textContent = "AI Model Failed";
    document.getElementById('loaderSubtext').textContent = "Please connect to the internet to load ML models.";
  }
}

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true });
    video.srcObject = stream;
    await video.play();
    initAudioMeter(stream);
    bootML();
  } catch (err) {
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('loaderText').textContent = 'Sensors Paused';
    document.getElementById('loaderSubtext').textContent = 'Please tap below to initialize Camera & Microphone.';
    const btn = document.getElementById('startBtn');
    btn.style.display = 'inline-block';
    btn.addEventListener('click', () => {
      btn.style.display = 'none';
      document.getElementById('spinner').style.display = 'block';
      initCamera();
    });
  }
}

initCamera();