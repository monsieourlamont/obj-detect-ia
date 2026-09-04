(function initBackground(){
    const bg = document.getElementById('bgCanvas');
    const bgCtx = bg.getContext('2d');
    const cell = 42; // debe coincidir con background-size del CSS
    let cols, rows, glowCells;

    function resize(){
      bg.width = window.innerWidth;
      bg.height = window.innerHeight;
      cols = Math.ceil(bg.width / cell);
      rows = Math.ceil(bg.height / cell);
      glowCells = [];
    }

    function spawnGlow(){
      // probabilidad de que se encienda una nueva celda en este frame
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
      // Transparente: la grid de fondo la pinta el CSS del propio canvas
      bgCtx.clearRect(0, 0, bg.width, bg.height);

      spawnGlow();
      glowCells = glowCells.filter(c => c.age < c.life);

      glowCells.forEach(c => {
        const progress = c.age / c.life;
        const alpha = Math.sin(progress * Math.PI) * 0.4; // entra y se apaga suavemente
        const x = c.col * cell;
        const y = c.row * cell;

        bgCtx.fillStyle = `rgba(57,255,136,${alpha * 0.5})`;
        bgCtx.fillRect(x, y, cell, cell);

        bgCtx.strokeStyle = `rgba(57,255,136,${alpha})`;
        bgCtx.lineWidth = 1;
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
    loop();
  })();
