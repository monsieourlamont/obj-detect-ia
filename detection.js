// ==============================
// detection.js - YOLO26n (ONNX, NMS-free)
// ==============================

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const viewport = document.getElementById('viewport');
const statusText = document.getElementById('statusText');
const objectCount = document.getElementById('objectCount');
const toggleBtn = document.getElementById('toggleBtn');
const flipBtn = document.getElementById('flipBtn');
const navDot = document.getElementById('navDot');
const navStatusLabel = document.getElementById('navStatusLabel');

let session = null;
let currentStream = null;
let cameraActive = false;
let detecting = false;
let rafId = null;
let facingMode = 'environment';

// COCO 80 clases (en español)
const COCO_CLASSES = [
  'persona','bicicleta','auto','motocicleta','avion','autobus','tren','camion','barco','semaforo',
  'hidrante','senal_pare','parquimetro','banco','pajaro','gato','perro','caballo','oveja','vaca',
  'elefante','oso','cebra','jirafa','mochila','paraguas','bolso','corbata','maleta','frisbee',
  'esquis','snowboard','pelota','cometa','bate','guante','patineta','tabla_surf','raqueta','botella',
  'copa_vino','taza','tenedor','cuchillo','cuchara','tazon','banana','manzana','sandwich','naranja',
  'brocoli','zanahoria','hot_dog','pizza','dona','pastel','silla','sofa','maceta','cama',
  'mesa','inodoro','monitor','laptop','mouse','control','teclado','celular','microondas','horno',
  'tostadora','lavamanos','refrigerador','libro','reloj','jarron','tijeras','oso_peluche','secador','cepillo'
];

const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.4;

function setStatus(msg, isError = false) {
  statusText.textContent = msg;
  statusText.classList.toggle('error', isError);
}

function setNav(label, on) {
  navStatusLabel.textContent = label;
  navDot.classList.toggle('on', on);
}

function updateButton() {
  if (cameraActive) {
    toggleBtn.textContent = 'Desactivar cámara';
    toggleBtn.classList.add('active');
  } else {
    toggleBtn.textContent = 'Activar cámara';
    toggleBtn.classList.remove('active');
  }
}

// ---------- Cargar modelo YOLO26n ----------
async function loadModel() {
  try {
    setStatus('Cargando YOLO26n...');
    setNav('CARGANDO MODELO', false);

    if (typeof ort === 'undefined') {
      throw new Error('onnxruntime-web no cargó. Revisa el script en index.html');
    }

    // El archivo debe estar en la raíz del sitio
    const modelUrl = '/yolo26n.onnx';

    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    setStatus('Modelo listo. Pulsa "Activar cámara" para comenzar.');
    setNav('MODELO LISTO', true);
    toggleBtn.disabled = false;
    toggleBtn.textContent = 'Activar cámara';
  } catch (err) {
    console.error(err);
    setStatus('ERROR: no se pudo cargar YOLO26n. ¿Subiste yolo26n.onnx?', true);
    setNav('ERROR DE MODELO', false);
  }
}

// ---------- Preprocesado (letterbox 640x640) ----------
function preprocess(videoEl) {
  const srcW = videoEl.videoWidth;
  const srcH = videoEl.videoHeight;

  const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = (INPUT_SIZE - newW) / 2;
  const padY = (INPUT_SIZE - newH) / 2;

  const off = document.createElement('canvas');
  off.width = INPUT_SIZE;
  off.height = INPUT_SIZE;
  const octx = off.getContext('2d');
  octx.fillStyle = '#000';
  octx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  octx.drawImage(videoEl, padX, padY, newW, newH);

  const imgData = octx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = imgData;

  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    float32[i] = data[i * 4] / 255;
    float32[INPUT_SIZE * INPUT_SIZE + i] = data[i * 4 + 1] / 255;
    float32[2 * INPUT_SIZE * INPUT_SIZE + i] = data[i * 4 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale,
    padX,
    padY,
    srcW,
    srcH
  };
}

// ---------- Postprocesado YOLO26 (NMS-free) ----------
// Salida típica: [1, 300, 6] → [x1, y1, x2, y2, score, class_id]
function postprocess(output, meta) {
  const data = output.data;
  const dims = output.dims;

  let detections = [];

  // Caso NMS-free: [1, N, 6]
  if (dims.length === 3 && dims[2] === 6) {
    const num = dims[1];
    for (let i = 0; i < num; i++) {
      const base = i * 6;
      const score = data[base + 4];
      if (score < CONF_THRESHOLD) continue;

      let x1 = data[base + 0];
      let y1 = data[base + 1];
      let x2 = data[base + 2];
      let y2 = data[base + 3];
      const classId = Math.round(data[base + 5]);

      // Quitar padding y escalar al tamaño del video
      x1 = (x1 - meta.padX) / meta.scale;
      y1 = (y1 - meta.padY) / meta.scale;
      x2 = (x2 - meta.padX) / meta.scale;
      y2 = (y2 - meta.padY) / meta.scale;

      detections.push({
        x1, y1, x2, y2,
        score,
        classId,
        label: COCO_CLASSES[classId] || `clase_${classId}`
      });
    }
    return detections;
  }

  // Fallback por si el export viene en formato antiguo [1, 84, 8400]
  if (dims.length === 3 && dims[1] === 84) {
    const numBoxes = dims[2];
    const boxes = [];
    for (let i = 0; i < numBoxes; i++) {
      let maxScore = 0;
      let classId = 0;
      for (let c = 0; c < 80; c++) {
        const s = data[(4 + c) * numBoxes + i];
        if (s > maxScore) {
          maxScore = s;
          classId = c;
        }
      }
      if (maxScore < CONF_THRESHOLD) continue;

      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w  = data[2 * numBoxes + i];
      const h  = data[3 * numBoxes + i];

      let x1 = (cx - w / 2 - meta.padX) / meta.scale;
      let y1 = (cy - h / 2 - meta.padY) / meta.scale;
      let x2 = (cx + w / 2 - meta.padX) / meta.scale;
      let y2 = (cy + h / 2 - meta.padY) / meta.scale;

      boxes.push({ x1, y1, x2, y2, score: maxScore, classId, label: COCO_CLASSES[classId] || `clase_${classId}` });
    }
    return boxes;
  }

  console.warn('Formato de salida no reconocido:', dims);
  return [];
}

// ---------- Cámara ----------
async function startCamera() {
  toggleBtn.disabled = true;
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode },
      audio: false
    });
    video.srcObject = currentStream;
    await new Promise(r => { video.onloadedmetadata = r; });
    await video.play();

    cameraActive = true;
    viewport.classList.remove('idle');
    updateButton();
    setNav('DETECTANDO', true);
    setStatus('Detectando objetos...');

    detecting = true;
    detectLoop();
  } catch (err) {
    console.error(err);
    let msg = 'No se pudo acceder a la cámara.';
    if (err.name === 'NotAllowedError') msg = 'Permiso de cámara denegado.';
    else if (err.name === 'NotFoundError') msg = 'No se detectó cámara.';
    setStatus('ERROR: ' + msg, true);
    setNav('SIN CÁMARA', false);
    cameraActive = false;
    viewport.classList.add('idle');
    updateButton();
  } finally {
    toggleBtn.disabled = false;
    flipBtn.disabled = false;
  }
}

function stopCamera() {
  detecting = false;
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  video.pause();
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  cameraActive = false;
  viewport.classList.add('idle');
  updateButton();
  setNav('EN ESPERA', false);
  setStatus('Cámara desactivada.');
  objectCount.textContent = '0 objetos';
}

function updateFlipLabel() {
  flipBtn.title = facingMode === 'environment'
    ? 'Cambiar a cámara frontal'
    : 'Cambiar a cámara trasera';
}

async function flipCamera() {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  updateFlipLabel();
  if (!cameraActive) return;

  flipBtn.disabled = true;
  detecting = false;
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  setStatus('Cambiando de cámara...');
  await startCamera();
}

toggleBtn.addEventListener('click', () => {
  if (cameraActive) stopCamera();
  else startCamera();
});
flipBtn.addEventListener('click', flipCamera);
updateFlipLabel();

// ---------- Bucle de detección ----------
async function detectLoop() {
  if (!detecting || !session) return;

  try {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const meta = preprocess(video);
    const results = await session.run({ images: meta.tensor });
    const output = results[Object.keys(results)[0]];
    const detections = postprocess(output, meta);

    if (!detecting) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(d => {
      const x = d.x1;
      const y = d.y1;
      const w = d.x2 - d.x1;
      const h = d.y2 - d.y1;

      ctx.strokeStyle = '#39ff88';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const label = `${d.label} ${Math.round(d.score * 100)}%`;
      ctx.font = '600 14px "JetBrains Mono", monospace';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = 'rgba(6,18,10,0.85)';
      ctx.fillRect(x - 1, y > 20 ? y - 20 : y, textWidth + 8, 18);

      ctx.fillStyle = '#39ff88';
      ctx.fillText(label, x + 3, y > 20 ? y - 6 : y + 13);
    });

    objectCount.textContent = `${detections.length} objeto${detections.length === 1 ? '' : 's'}`;
  } catch (err) {
    console.error('Error en detección:', err);
  }

  rafId = requestAnimationFrame(detectLoop);
}

// ---------- Inicio ----------
loadModel();
