/* neural.js — Animated neural network background */
(function () {
  const canvas = document.getElementById('neuralCanvas');
  const ctx = canvas.getContext('2d');

  let W, H, nodes = [], edges = [];
  const NODE_COUNT = 55;
  const SPEED = 0.3;
  const CONNECT_DIST = 130;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function randomNode() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * SPEED,
      vy: (Math.random() - 0.5) * SPEED,
      r: Math.random() * 2 + 1,
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 0.02 + Math.random() * 0.03,
    };
  }

  function init() {
    resize();
    nodes = Array.from({ length: NODE_COUNT }, randomNode);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < CONNECT_DIST) {
          const alpha = (1 - d / CONNECT_DIST) * 0.4;
          ctx.strokeStyle = `rgba(79,142,247,${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }

    // Nodes
    nodes.forEach(n => {
      n.pulse += n.pulseSpeed;
      const glow = 0.6 + Math.sin(n.pulse) * 0.4;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r * glow, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(79,142,247,${glow * 0.8})`;
      ctx.fill();
    });
  }

  function move() {
    nodes.forEach(n => {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });
  }

  function loop() {
    move(); draw();
    requestAnimationFrame(loop);
  }

  init();
  loop();
  window.addEventListener('resize', () => { resize(); });
})();
