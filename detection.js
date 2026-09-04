const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const viewport = document.getElementById('viewport');
  const idleMsg = document.getElementById('idleMsg');
  const statusText = document.getElementById('statusText');
  const objectCount = document.getElementById('objectCount');
  const toggleBtn = document.getElementById('toggleBtn');
  const flipBtn = document.getElementById('flipBtn');
  const navDot = document.getElementById('navDot');
  const navStatusLabel = document.getElementById('navStatusLabel');

  let model = null;
  let currentStream = null;
  let cameraActive = false;
  let detecting = false;
  let rafId = null;
  let facingMode = 'environment'; // 'environment' = trasera, 'user' = frontal

  // ---------- Fondo animado (grid con destellos aleatorios) ----------

function setStatus(msg, isError = false){
    statusText.textContent = msg;
    statusText.classList.toggle('error', isError);
  }

  function setNav(label, on){
    navStatusLabel.textContent = label;
    navDot.classList.toggle('on', on);
  }

  function updateButton(){
    if (cameraActive){
      toggleBtn.textContent = 'Desactivar cámara';
      toggleBtn.classList.add('active');
    } else {
      toggleBtn.textContent = 'Activar cámara';
      toggleBtn.classList.remove('active');
    }
  }

  // ---------- Cargar modelo ----------
  async function loadModel(){
    try{
      setStatus('Cargando modelo de IA...');
      model = await cocoSsd.load();
      setStatus('Modelo listo. Pulsa "Activar cámara" para comenzar.');
      setNav('MODELO LISTO', true);
      toggleBtn.disabled = false;
      toggleBtn.textContent = 'Activar cámara';
    } catch(err){
      console.error(err);
      setStatus('ERROR: no se pudo cargar el modelo de IA.', true);
      setNav('ERROR DE MODELO', false);
    }
  }

  // ---------- Cámara ----------
  async function startCamera(){
    toggleBtn.disabled = true;
    try{
      currentStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode },
        audio: false
      });
      video.srcObject = currentStream;

      await new Promise(resolve => { video.onloadedmetadata = resolve; });
      await video.play();

      cameraActive = true;
      viewport.classList.remove('idle');
      updateButton();
      setNav('DETECTANDO', true);
      setStatus('Detectando objetos...');

      detecting = true;
      detectLoop();
    } catch(err){
      console.error(err);
      let msg = 'No se pudo acceder a la cámara.';
      if (err.name === 'NotAllowedError') msg = 'Permiso de cámara denegado. Actívalo en los ajustes del navegador.';
      else if (err.name === 'NotFoundError') msg = 'No se detectó una cámara con esa orientación en este dispositivo.';
      else if (location.protocol !== 'https:' && location.hostname !== 'localhost') msg = 'Esta página requiere HTTPS para usar la cámara.';
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

  function stopCamera(){
    detecting = false;
    if (rafId){
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (currentStream){
      currentStream.getTracks().forEach(track => track.stop());
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

  function updateFlipLabel(){
    flipBtn.title = facingMode === 'environment'
      ? 'Cambiar a cámara frontal'
      : 'Cambiar a cámara trasera';
  }

  async function flipCamera(){
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    updateFlipLabel();

    // Si la cámara no está activa, solo guardamos la preferencia para la próxima vez
    if (!cameraActive) return;

    flipBtn.disabled = true;
    detecting = false;
    if (rafId){
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    if (currentStream){
      currentStream.getTracks().forEach(track => track.stop());
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
  async function detectLoop(){
    if (!detecting || !model) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const predictions = await model.detect(video);
    if (!detecting) return; // la cámara pudo apagarse mientras se detectaba

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let visible = 0;
    predictions.forEach(pred => {
      if (pred.score > 0.6){
        visible++;
        const [x, y, width, height] = pred.bbox;

        ctx.strokeStyle = '#39ff88';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        const label = `${pred.class} ${Math.round(pred.score * 100)}%`;
        ctx.font = '600 14px "JetBrains Mono", monospace';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = 'rgba(6,18,10,0.85)';
        ctx.fillRect(x - 1, y > 20 ? y - 20 : y, textWidth + 8, 18);

        ctx.fillStyle = '#39ff88';
        ctx.fillText(label, x + 3, y > 20 ? y - 6 : y + 13);
      }
    });

    objectCount.textContent = `${visible} objeto${visible === 1 ? '' : 's'}`;

    rafId = requestAnimationFrame(detectLoop);
  }

  // ---------- Inicio ----------
  loadModel();
