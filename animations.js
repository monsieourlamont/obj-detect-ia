// ==============================
// animations.js - Fondo con grid alineado (PC + móvil)
// ==============================
(function initBackground(){
  const bg = document.getElementById('bgCanvas');
  if (!bg) return;

  const bgCtx = bg.getContext('2d');
  const cell = 42; // debe coincidir con background-size del CSS
  let cols = 0;
  let rows = 0;
  let glowCells = [];

  function resize(){
    // Usar el tamaño REAL del elemento (evita el desfase por scrollbar en PC)
    const w = Math.max(1, bg.clientWidth || window.innerWidth);
    const h = Math.max(1, bg.clientHeight || window.innerHeight);

    // Resolución interna = tamaño CSS (1 unidad de canvas = 1 px CSS)
    bg.width = w;
    bg.height = h;

    cols = Math.ceil(w / cell);
    rows = Math.ceil(h / cell);
    glowCells = [];
  }

  function spawnGlow(){
    if (Math.random() < 0.05){
      glowCells.push({
        col: Math.floor(Math.random() * cols),
        row: Math.floor(Math.random() * rows),
        age: 0,
        life: 45 + Math.random() * 70
      });
    }
  }

  function drawFrame(){
    bgCtx.clearRect(0, 0, bg.width, bg.height);

    spawnGlow();
    glowCells = glowCells.filter(c => c.age < c.life);

    glowCells.forEach(c => {
      const progress = c.age / c.life;
      const alpha = Math.sin(progress * Math.PI) * 0.4;
      // Posición exacta en múltiplos de cell (alineada con el CSS)
      const x = c.col * cell;
      const y = c.row * cell;

      bgCtx.fillStyle = `rgba(57,255,136,${alpha * 0.5})`;
      bgCtx.fillRect(x, y, cell, cell);

      bgCtx.strokeStyle = `rgba(57,255,136,${alpha})`;
      bgCtx.lineWidth = 1;
      // +0.5 para líneas nítidas de 1px
      bgCtx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);

      c.age++;
    });
  }

  function loop(){
    drawFrame();
    requestAnimationFrame(loop);
  }

  resize();
  window.addEventListener('resize', resize);
  // Por si el layout cambia al cargar fuentes / scrollbar
  window.addEventListener('load', resize);
  loop();
})();
