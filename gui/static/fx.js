/*
 * Visual effects for sapai arena: particle bursts, flying coins, floating text,
 * and the painted battle backdrop. Everything is drawn on canvas; no images.
 * Respects prefers-reduced-motion (bursts are skipped, text still shows).
 */
window.FX = (() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  const cv = document.createElement("canvas");
  cv.id = "fx";
  cv.setAttribute("aria-hidden", "true");
  Object.assign(cv.style, { position: "fixed", inset: "0", width: "100%", height: "100%", pointerEvents: "none", zIndex: "30" });
  document.body.appendChild(cv);
  const g = cv.getContext("2d");
  let W = 0, H = 0, DPR = 1;
  function resize() {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    W = innerWidth; H = innerHeight;
    cv.width = W * DPR; cv.height = H * DPR;
    g.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  resize();
  addEventListener("resize", resize);

  const css = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const parts = [], texts = [];
  let running = false;
  const rnd = (a, b) => a + Math.random() * (b - a);

  function at(x) {
    if (!x) return { x: W / 2, y: H / 2 };
    if (x.getBoundingClientRect) x = x.getBoundingClientRect();
    if ("width" in x) return { x: x.left + x.width / 2, y: x.top + x.height / 2 };
    return x;
  }

  const KINDS = {
    sparkle: { n: 12, colors: () => [css("--accent"), css("--gold"), "#ffffff"], shape: "star", speed: [1.5, 4], life: [28, 44], size: [3, 6], grav: 0.02 },
    coins:   { n: 10, colors: () => [css("--gold"), "#ffd54a"], shape: "coin", speed: [2, 5], life: [34, 52], size: [4, 6], grav: 0.16 },
    frost:   { n: 16, colors: () => [css("--freeze"), "#cfe6ff", "#ffffff"], shape: "shard", speed: [1.5, 4.5], life: [30, 50], size: [3, 7], grav: 0.01 },
    crumbs:  { n: 12, colors: () => ["#d9a25b", "#b97a3a", "#f1d29a"], shape: "crumb", speed: [1.5, 4], life: [26, 40], size: [2, 4], grav: 0.2 },
    hearts:  { n: 6, colors: () => [css("--hp"), "#ff8fa3"], shape: "heart", speed: [0.6, 1.6], life: [40, 60], size: [5, 8], grav: -0.04 },
    stars:   { n: 26, colors: () => [css("--gold"), "#ffe27a", css("--accent"), "#ffffff"], shape: "star", speed: [2, 7], life: [40, 70], size: [4, 8], grav: 0.06 },
    poof:    { n: 14, colors: () => ["#9aa3a0", "#c9cfcc", "#7b8480"], shape: "puff", speed: [0.6, 2.4], life: [30, 48], size: [6, 13], grav: -0.03 },
    confetti:{ n: 70, colors: () => [css("--accent"), css("--gold"), css("--hp"), css("--freeze"), css("--atk")], shape: "confetti", speed: [4, 10], life: [80, 130], size: [4, 8], grav: 0.18 },
    hit:     { n: 8, colors: () => ["#ffffff", css("--atk"), css("--hp")], shape: "shard", speed: [2.5, 5], life: [14, 22], size: [3, 6], grav: 0 },
  };

  function burst(kind, where, opts = {}) {
    if (reduce.matches) return;
    const k = KINDS[kind];
    if (!k) return;
    const p = at(where), cols = k.colors();
    const n = Math.round((opts.n || k.n) * (opts.scale || 1));
    for (let i = 0; i < n; i++) {
      const a = opts.up ? rnd(-Math.PI * 0.85, -Math.PI * 0.15) : rnd(0, Math.PI * 2);
      const s = rnd(...k.speed) * (opts.power || 1);
      parts.push({ x: p.x + rnd(-6, 6), y: p.y + rnd(-6, 6), vx: Math.cos(a) * s, vy: Math.sin(a) * s,
        life: 0, max: rnd(...k.life), size: rnd(...k.size), color: cols[i % cols.length],
        shape: k.shape, grav: k.grav, rot: rnd(0, 6.28), vr: rnd(-0.3, 0.3) });
    }
    start();
  }

  // Coins (or another shape) that fly from one place to another along an arc
  function fly(from, to, opts = {}) {
    if (reduce.matches) return;
    const a = at(from), b = at(to);
    const n = opts.n || 6, cols = (KINDS[opts.kind || "coins"]).colors();
    for (let i = 0; i < n; i++) {
      parts.push({ fly: true, ax: a.x + rnd(-10, 10), ay: a.y + rnd(-10, 10), bx: b.x, by: b.y,
        cx: (a.x + b.x) / 2 + rnd(-60, 60), cy: Math.min(a.y, b.y) - rnd(40, 120),
        life: -i * 3, max: 34, size: 5, color: cols[i % cols.length], shape: KINDS[opts.kind || "coins"].shape,
        rot: 0, vr: 0.2, done: opts.onArrive && i === n - 1 ? opts.onArrive : null });
    }
    start();
  }

  function text(where, str, opts = {}) {
    const p = at(where);
    texts.push({ x: p.x + (opts.dx || 0), y: p.y + (opts.dy || 0), str, color: opts.color || css("--fg"),
      size: opts.size || 18, life: 0, max: opts.max || 55, rise: opts.rise ?? 0.9, pop: true });
    start();
  }

  function start() {
    if (!running) { running = true; requestAnimationFrame(tick); }
  }

  function drawShape(p, alpha) {
    g.save();
    g.globalAlpha = alpha;
    g.translate(p.x, p.y);
    g.rotate(p.rot);
    g.fillStyle = p.color;
    const s = p.size;
    switch (p.shape) {
      case "coin":
        g.beginPath(); g.ellipse(0, 0, s, s * Math.abs(Math.cos(p.rot * 2)) + 1, 0, 0, 6.28); g.fill();
        g.fillStyle = "rgba(255,255,255,.55)"; g.beginPath(); g.ellipse(-s * 0.3, -s * 0.3, s * 0.3, s * 0.2, 0, 0, 6.28); g.fill();
        break;
      case "star":
        g.beginPath();
        for (let i = 0; i < 10; i++) { const r = i % 2 ? s * 0.45 : s; g.lineTo(Math.cos(i * 0.628) * r, Math.sin(i * 0.628) * r); }
        g.closePath(); g.fill();
        break;
      case "shard":
        g.beginPath(); g.moveTo(0, -s * 1.4); g.lineTo(s * 0.35, 0); g.lineTo(0, s * 1.4); g.lineTo(-s * 0.35, 0); g.closePath(); g.fill();
        break;
      case "crumb":
        g.fillRect(-s / 2, -s / 2, s, s * 0.8);
        break;
      case "heart":
        g.rotate(-p.rot);
        g.beginPath(); g.moveTo(0, s * 0.9);
        g.bezierCurveTo(-s * 1.4, -s * 0.2, -s * 0.6, -s * 1.3, 0, -s * 0.4);
        g.bezierCurveTo(s * 0.6, -s * 1.3, s * 1.4, -s * 0.2, 0, s * 0.9); g.fill();
        break;
      case "puff":
        g.beginPath(); g.arc(0, 0, s * (0.6 + p.life / p.max), 0, 6.28); g.fill();
        break;
      case "confetti":
        g.fillRect(-s / 2, -s / 4, s, s / 2);
        break;
    }
    g.restore();
  }

  function tick() {
    g.clearRect(0, 0, W, H);
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      p.life++;
      if (p.fly) {
        if (p.life < 0) continue;
        const t = Math.min(1, p.life / p.max), e = 1 - Math.pow(1 - t, 2);
        p.x = (1 - e) * (1 - e) * p.ax + 2 * (1 - e) * e * p.cx + e * e * p.bx;
        p.y = (1 - e) * (1 - e) * p.ay + 2 * (1 - e) * e * p.cy + e * e * p.by;
        p.rot += p.vr;
        drawShape(p, 1);
        if (t >= 1) { if (p.done) p.done(); parts.splice(i, 1); }
        continue;
      }
      p.x += p.vx; p.y += p.vy; p.vy += p.grav; p.vx *= 0.97; p.vy *= p.shape === "confetti" ? 0.99 : 0.97;
      p.rot += p.vr;
      const fade = 1 - p.life / p.max;
      if (fade <= 0) { parts.splice(i, 1); continue; }
      drawShape(p, p.shape === "puff" ? fade * 0.6 : Math.min(1, fade * 1.6));
    }
    for (let i = texts.length - 1; i >= 0; i--) {
      const t = texts[i];
      t.life++;
      const k = t.life / t.max;
      if (k >= 1) { texts.splice(i, 1); continue; }
      const scale = t.life < 8 ? 0.6 + (t.life / 8) * 0.55 : 1.15 - Math.min(0.15, (t.life - 8) / 40);
      t.y -= t.rise;
      g.save();
      g.globalAlpha = k > 0.7 ? (1 - k) / 0.3 : 1;
      g.font = `800 ${t.size * scale}px "Bricolage Grotesque", system-ui, sans-serif`;
      g.textAlign = "center"; g.textBaseline = "middle";
      g.lineWidth = 4; g.strokeStyle = css("--surface", "#fff");
      g.strokeText(t.str, t.x, t.y);
      g.fillStyle = t.color; g.fillText(t.str, t.x, t.y);
      g.restore();
    }
    if (parts.length || texts.length) requestAnimationFrame(tick);
    else { running = false; g.clearRect(0, 0, W, H); }
  }

  // Restart a CSS animation class on an element
  function anim(el, cls, ms = 600) {
    if (!el) return;
    el.classList.remove(cls);
    void el.offsetWidth;
    el.classList.add(cls);
    setTimeout(() => el.classList.remove(cls), ms);
  }

  // ------------------------------------------------------- battle backdrop
  // A painted meadow: sky, sun or moon, drifting clouds, two hill layers, grass.
  const scene = { el: null, raf: 0, t: 0, clouds: null, tufts: null };
  function drawScene() {
    const c = scene.el;
    if (!c || !c.isConnected) return;
    const r = c.getBoundingClientRect();
    if (!r.width) return;
    const d = Math.min(2, devicePixelRatio || 1);
    if (c.width !== Math.round(r.width * d) || c.height !== Math.round(r.height * d)) {
      c.width = Math.round(r.width * d); c.height = Math.round(r.height * d);
    }
    const x = c.getContext("2d"), w = r.width, h = r.height;
    x.setTransform(d, 0, 0, d, 0, 0);
    const dark = css("color-scheme") === "dark" || getComputedStyle(document.documentElement).colorScheme === "dark";
    const sky = x.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, css("--sky-top")); sky.addColorStop(1, css("--sky-bottom"));
    x.fillStyle = sky; x.fillRect(0, 0, w, h);
    // sun or moon
    x.save();
    x.fillStyle = css("--sun");
    x.shadowColor = css("--sun"); x.shadowBlur = 30;
    x.beginPath(); x.arc(w * 0.82, h * 0.24, Math.min(26, h * 0.12), 0, 6.28); x.fill();
    if (dark) { x.shadowBlur = 0; x.fillStyle = css("--sky-top"); x.beginPath(); x.arc(w * 0.82 + 10, h * 0.24 - 6, Math.min(22, h * 0.1), 0, 6.28); x.fill(); }
    x.restore();
    if (dark) {
      x.fillStyle = "rgba(255,255,255,.7)";
      for (let i = 0; i < 40; i++) { const sx = (i * 97.3) % w, sy = (i * 41.7) % (h * 0.5); x.fillRect(sx, sy, 1.5, 1.5); }
    }
    // clouds
    if (!scene.clouds) scene.clouds = Array.from({ length: 5 }, (_, i) => ({ x: i * 0.23 + Math.random() * 0.1, y: 0.1 + Math.random() * 0.3, s: 0.6 + Math.random() * 0.6, v: 0.00004 + Math.random() * 0.00005 }));
    x.fillStyle = css("--cloud");
    for (const cl of scene.clouds) {
      const cx = (((cl.x + scene.t * cl.v) % 1.3) - 0.15) * w, cy = cl.y * h, s = cl.s * 22;
      x.beginPath();
      x.arc(cx, cy, s, 0, 6.28); x.arc(cx + s * 1.1, cy + s * 0.2, s * 0.8, 0, 6.28);
      x.arc(cx - s * 1.1, cy + s * 0.25, s * 0.7, 0, 6.28); x.arc(cx + s * 0.2, cy - s * 0.5, s * 0.75, 0, 6.28);
      x.fill();
    }
    // hills
    const hill = (base, amp, freq, phase, color) => {
      x.fillStyle = color; x.beginPath(); x.moveTo(0, h);
      for (let i = 0; i <= w; i += 8) x.lineTo(i, h * base - Math.sin(i * freq + phase) * amp - Math.sin(i * freq * 2.3 + phase) * amp * 0.3);
      x.lineTo(w, h); x.closePath(); x.fill();
    };
    hill(0.66, 14, 0.012, 1.3, css("--hill-far"));
    hill(0.76, 10, 0.02, 4.1, css("--hill-near"));
    // ground with grass tufts
    const gy = h * 0.82;
    x.fillStyle = css("--ground"); x.fillRect(0, gy, w, h - gy);
    if (!scene.tufts) scene.tufts = Array.from({ length: 60 }, () => [Math.random(), Math.random()]);
    x.strokeStyle = css("--hill-near"); x.lineWidth = 1.5;
    for (const [tx, ty] of scene.tufts) {
      const px = tx * w, py = gy + ty * (h - gy) * 0.8 + 3;
      x.beginPath(); x.moveTo(px, py); x.lineTo(px - 2, py - 5); x.moveTo(px, py); x.lineTo(px + 2, py - 6); x.stroke();
    }
  }
  function sceneLoop(now) {
    scene.t = now;
    drawScene();
    if (!reduce.matches && scene.el && scene.el.isConnected && scene.el.offsetParent) scene.raf = requestAnimationFrame(sceneLoop);
    else scene.raf = 0;
  }
  function attachScene(el) {
    scene.el = el;
    if (!scene.raf) scene.raf = requestAnimationFrame(sceneLoop);
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawScene);
  addEventListener("resize", drawScene);

  return { burst, fly, text, anim, at, attachScene, drawScene };
})();
