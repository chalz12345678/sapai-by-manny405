/*
 * 3D battle view for sapai arena (prototype, "option A": cardboard standees).
 *
 * Each pet is a cardboard standee: a card with its emoji art, a thickness, a
 * fold-out stand and a shadow, on a small 3D meadow. The replay frames from the
 * server drive everything: every step is animated from what changed between
 * two frames plus the step's events (who acted on whom):
 *   attack   front pets lunge, collide, sparks and camera shake
 *   ability  the acting pet glows and an orb flies to each target
 *   damage   red flash and a floating number
 *   buff     green sparkles and a floating +x/+y
 *   faint    the standee tips over backwards and sinks away
 *   summon   a new standee pops up
 * Three.js is loaded from cdnjs only when the 3D view is first opened.
 */
window.Battle3D = (() => {
  const THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
  let loading = null;
  function load() {
    if (window.THREE) return Promise.resolve();
    if (!loading) {
      loading = new Promise((ok, fail) => {
        const s = document.createElement("script");
        s.src = THREE_URL;
        s.onload = () => (window.THREE ? ok() : fail(new Error("Three.js did not load")));
        s.onerror = () => { loading = null; fail(new Error("Could not load the 3D library. 3D needs an internet connection the first time.")); };
        document.head.appendChild(s);
      });
    }
    return loading;
  }

  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  const css = (v, fb) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || fb;
  const SPACING = 1.38, GAP = 0.95;
  const slotX = (side, i) => (side === 0 ? -1 : 1) * (GAP + i * SPACING);

  let T, renderer, scene, camera, root, el, ro, raf = 0;
  let pets = [[], []];       // current standees per side, front first
  let tweens = [], fx = [];
  let shake = 0, camBase = null, mounted = false, lastFrame = null;
  // camera director: eases toward a focus point and zoom, then back to the wide shot
  const cam = { x: 0, zoom: 1, tx: 0, tzoom: 1, until: 0 };
  function focus(x, zoom, holdMs) {
    if (reduce.matches) return;
    cam.tx = Math.max(-5, Math.min(5, x)); cam.tzoom = zoom; cam.until = performance.now() + holdMs;
  }

  // ------------------------------------------------------------- textures
  function roundRect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }
  function tierColors(tier) {
    const k = Number.isInteger(tier) ? tier : 0;
    return [css(k ? `--t${k}` : "--sunk", "#e2e7e0"), css(k ? `--t${k}e` : "--muted", "#5d6b63")];
  }
  function cardTexture(p, flip) {
    const c = document.createElement("canvas"); c.width = 400; c.height = 520;
    const g = c.getContext("2d");
    const [bg, edge] = tierColors(p.tier);
    g.fillStyle = "#fbfcfa"; roundRect(g, 6, 6, 388, 508, 36); g.fill();
    g.lineWidth = 8; g.strokeStyle = "#d7ddd5"; g.stroke();
    g.fillStyle = edge; roundRect(g, 6, 6, 388, 30, 18); g.fill();
    g.fillStyle = bg; g.beginPath(); g.ellipse(200, 230, 150, 140, 0, 0, 6.283); g.fill();
    g.fillStyle = "rgba(0,0,0,.14)"; g.beginPath(); g.ellipse(200, 340, 90, 16, 0, 0, 6.283); g.fill();
    g.save(); g.translate(200, 230);
    g.fillStyle = "#000"; // emoji inherit the fill's opacity, so reset it after the shadow
    if (flip) g.scale(-1, 1);
    g.font = '190px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif';
    g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(p.emoji, 0, 6);
    g.restore();
    g.fillStyle = "#1b2420"; g.font = '700 44px "Bricolage Grotesque", system-ui, sans-serif';
    g.textAlign = "center"; g.fillText(cap(p.name), 200, 440);
    g.fillStyle = "#5d6b63"; g.font = '600 26px "JetBrains Mono", monospace';
    g.fillText(Number.isInteger(p.tier) ? `TIER ${p.tier}` : "SUMMONED", 200, 484);
    const t = new T.CanvasTexture(c); t.encoding = T.sRGBEncoding; t.anisotropy = 4;
    return t;
  }
  const cap = (s) => String(s).replace(/\b\w/g, (m) => m.toUpperCase());
  function statsTexture(p) {
    const c = document.createElement("canvas"); c.width = 256; c.height = 96;
    const g = c.getContext("2d");
    const badge = (x, color, text, hex) => {
      g.fillStyle = color;
      if (hex) { g.beginPath(); g.moveTo(x + 14, 10); g.lineTo(x + 102, 10); g.lineTo(x + 118, 48); g.lineTo(x + 102, 86); g.lineTo(x + 14, 86); g.lineTo(x - 2, 48); g.closePath(); g.fill(); }
      else { roundRect(g, x, 10, 116, 76, 38); g.fill(); }
      g.fillStyle = "#fff"; g.font = '800 50px "JetBrains Mono", monospace'; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(text, x + 58, 50);
    };
    badge(6, css("--atk", "#b86a0b"), String(p.attack), true);
    badge(134, css("--hp", "#c0392f"), String(Math.max(p.health, 0)), false);
    if (p.status && p.status.emoji) { g.font = "40px sans-serif"; g.fillText(p.status.emoji, 128, 22); }
    const t = new T.CanvasTexture(c); t.encoding = T.sRGBEncoding; return t;
  }
  function textTexture(str, color) {
    const c = document.createElement("canvas"); c.width = 320; c.height = 120;
    const g = c.getContext("2d");
    g.font = '900 88px "Bricolage Grotesque", system-ui, sans-serif'; g.textAlign = "center"; g.textBaseline = "middle";
    g.lineWidth = 14; g.strokeStyle = "rgba(255,255,255,.95)"; g.strokeText(str, 160, 64);
    g.fillStyle = color; g.fillText(str, 160, 64);
    const t = new T.CanvasTexture(c); t.encoding = T.sRGBEncoding; return t;
  }
  let backTex = null;
  function backTexture() {
    if (backTex) return backTex;
    const c = document.createElement("canvas"); c.width = 400; c.height = 520;
    const g = c.getContext("2d");
    g.fillStyle = "#b88a5a"; g.fillRect(0, 0, 400, 520);
    for (let y = 0; y < 520; y += 6) { g.fillStyle = `rgba(90,60,30,${0.05 + Math.random() * 0.05})`; g.fillRect(0, y, 400, 2); }
    g.strokeStyle = "rgba(255,240,215,.55)"; g.lineWidth = 10; roundRect(g, 30, 30, 340, 460, 30); g.stroke();
    g.font = '170px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif';
    g.textAlign = "center"; g.textBaseline = "middle"; g.globalAlpha = 0.55; g.fillText("🐾", 200, 250);
    backTex = new T.CanvasTexture(c); backTex.encoding = T.sRGBEncoding;
    return backTex;
  }
  let glowTex = null;
  function glowTexture() {
    if (glowTex) return glowTex;
    const c = document.createElement("canvas"); c.width = c.height = 128;
    const g = c.getContext("2d"), gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.35, "rgba(255,255,255,.6)"); gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
    glowTex = new T.CanvasTexture(c); return glowTex;
  }

  // ------------------------------------------------------------- scene
  function buildScene() {
    scene = new T.Scene();
    const dark = getComputedStyle(document.documentElement).colorScheme === "dark";
    // sky
    const sc = document.createElement("canvas"); sc.width = 4; sc.height = 256;
    const sg = sc.getContext("2d"), grad = sg.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, css("--sky-top", "#9fd3f0")); grad.addColorStop(1, css("--sky-bottom", "#e4f3f7"));
    sg.fillStyle = grad; sg.fillRect(0, 0, 4, 256);
    const skyTex = new T.CanvasTexture(sc); skyTex.encoding = T.sRGBEncoding;
    scene.background = skyTex;
    scene.fog = new T.Fog(new T.Color(css("--sky-bottom", "#e4f3f7")), 30, 70);

    scene.add(new T.HemisphereLight(dark ? 0x6070a0 : 0xffffff, dark ? 0x1a2a20 : 0x6a8a50, dark ? 0.55 : 0.75));
    const sun = new T.DirectionalLight(dark ? 0xc8d4ff : 0xfff4dd, dark ? 0.55 : 0.95);
    sun.position.set(4, 9, 6); sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -9, right: 9, top: 6, bottom: -4, near: 1, far: 30 });
    sun.shadow.radius = 4;
    scene.add(sun);

    // ground with a little grass texture
    const gc = document.createElement("canvas"); gc.width = gc.height = 256;
    const gg = gc.getContext("2d");
    gg.fillStyle = css("--hill-near", "#7fbf6e"); gg.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 900; i++) {
      gg.fillStyle = `rgba(${dark ? "120,170,120" : "40,90,30"},${Math.random() * 0.18})`;
      gg.fillRect(Math.random() * 256, Math.random() * 256, 2, 5);
    }
    const gt = new T.CanvasTexture(gc); gt.wrapS = gt.wrapT = T.RepeatWrapping; gt.repeat.set(10, 10); gt.encoding = T.sRGBEncoding;
    const ground = new T.Mesh(new T.PlaneGeometry(80, 80), new T.MeshStandardMaterial({ map: gt, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
    scene.add(ground);
    // a dirt arena strip
    const strip = new T.Mesh(new T.CircleGeometry(1, 48), new T.MeshStandardMaterial({ color: new T.Color(css("--ground", "#c9a46a")), roughness: 1 }));
    strip.scale.set(8.2, 1.9, 1); strip.rotation.x = -Math.PI / 2; strip.position.y = 0.005; strip.receiveShadow = true;
    scene.add(strip);
    // hills and clouds
    const hillMat = new T.MeshLambertMaterial({ color: new T.Color(css("--hill-far", "#a9d39a")).multiplyScalar(0.85) });
    [[-14, -26, 8], [2, -30, 10], [17, -25, 7], [-30, -32, 10], [30, -30, 9]].forEach(([x, z, r]) => {
      const h = new T.Mesh(new T.SphereGeometry(r, 24, 12), hillMat); h.scale.y = 0.22; h.position.set(x, -0.3, z); scene.add(h);
    });
    const cloudMat = new T.MeshStandardMaterial({ color: dark ? 0x5a6488 : 0xffffff, roughness: 1, transparent: true, opacity: dark ? 0.5 : 0.95 });
    root.clouds = [];
    for (let k = 0; k < 5; k++) {
      const cl = new T.Group();
      for (let j = 0; j < 4; j++) {
        const b = new T.Mesh(new T.SphereGeometry(0.9 + Math.random() * 0.6, 14, 10), cloudMat);
        b.position.set(j * 1.1 - 1.6, Math.random() * 0.5, Math.random() * 0.4); cl.add(b);
      }
      cl.position.set(-22 + k * 10 + Math.random() * 4, 7 + Math.random() * 2.5, -16 - Math.random() * 6);
      cl.userData.v = 0.15 + Math.random() * 0.2;
      scene.add(cl); root.clouds.push(cl);
    }
    // sun or moon
    const orb = new T.Mesh(new T.SphereGeometry(1.4, 24, 16), new T.MeshBasicMaterial({ color: new T.Color(css("--sun", "#ffd35c")) }));
    orb.position.set(14, 11, -26); scene.add(orb);
    scene.add(root);
  }

  // ------------------------------------------------------------- standees
  function makeStandee(p, side) {
    const g = new T.Group();
    const flip = new T.Group(); // turns the card around (summons start face down)
    const lean = new T.Group(); // tips over on faint
    g.add(flip); flip.add(lean);
    const W = 1.05, H = 1.37, D = 0.05;
    // the printed face is unlit so the art keeps its real colors; the cardboard edges are lit
    const faceMat = new T.MeshBasicMaterial({ map: cardTexture(p, side === 0) });
    const cardboard = new T.MeshStandardMaterial({ color: 0xb88a5a, roughness: 0.95 });
    const backMat = new T.MeshStandardMaterial({ map: backTexture(), roughness: 0.95 });
    const card = new T.Mesh(new T.BoxGeometry(W, H, D), [cardboard, cardboard, cardboard, cardboard, faceMat, backMat]);
    card.position.y = H / 2 + 0.02; card.castShadow = true;
    lean.add(card);
    const stand = new T.Mesh(new T.BoxGeometry(0.34, 0.9, 0.03), cardboard);
    stand.position.set(0, 0.42, -0.25); stand.rotation.x = -0.45; stand.castShadow = true;
    lean.add(stand);
    const stats = new T.Sprite(new T.SpriteMaterial({ map: statsTexture(p), depthTest: false }));
    stats.scale.set(0.95, 0.36, 1); stats.position.set(0, -0.02, 0.12); stats.renderOrder = 5;
    g.add(stats);
    const glow = new T.Sprite(new T.SpriteMaterial({ map: glowTexture(), color: 0xffc94a, transparent: true, opacity: 0, depthWrite: false, blending: T.AdditiveBlending }));
    glow.scale.set(2.4, 2.4, 1); glow.position.set(0, 0.75, -0.1);
    g.add(glow);
    g.rotation.y = side === 0 ? 0.32 : -0.32;
    g.userData = { p: { ...p }, side, flip, lean, card, faceMat, stats, glow, dead: false, phase: Math.random() * 6.28 };
    return g;
  }
  function setStats(s, p) {
    const u = s.userData;
    if (u.p.attack === p.attack && u.p.health === p.health && (u.p.status || {}).name === (p.status || {}).name) return;
    u.stats.material.map.dispose();
    u.stats.material.map = statsTexture(p); u.stats.material.needsUpdate = true;
    u.p = { ...p };
  }
  function disposeStandee(s) {
    root.remove(s);
    s.traverse((o) => {
      if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => { if (m.map && m.map !== glowTex && m.map !== backTex) m.map.dispose(); m.dispose(); });
      if (o.geometry) o.geometry.dispose();
    });
  }

  // ------------------------------------------------------------- tweens & fx
  const ease = { out: (t) => 1 - Math.pow(1 - t, 3), inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2), back: (t) => 1 + 2.2 * Math.pow(t - 1, 3) + 1.2 * Math.pow(t - 1, 2) };
  function tween(delay, dur, fn, done) {
    if (reduce.matches) { dur = Math.min(dur, 1); }
    tweens.push({ t0: performance.now() + delay, dur: Math.max(1, dur), fn, done, started: false });
  }
  function floatText(pos, str, color) {
    const sp = new T.Sprite(new T.SpriteMaterial({ map: textTexture(str, color), depthTest: false, transparent: true }));
    sp.scale.set(1.1, 0.42, 1); sp.position.copy(pos); sp.renderOrder = 10;
    root.add(sp);
    fx.push({ obj: sp, life: 0, max: 70, vy: 0.012, kind: "text" });
  }
  function burst(pos, color, n = 14, speed = 0.06, size = 0.16, grav = -0.002) {
    if (reduce.matches) return;
    for (let i = 0; i < n; i++) {
      const sp = new T.Sprite(new T.SpriteMaterial({ map: glowTexture(), color, transparent: true, depthWrite: false, blending: T.AdditiveBlending }));
      const s = size * (0.6 + Math.random() * 0.8);
      sp.scale.set(s, s, 1); sp.position.copy(pos);
      root.add(sp);
      const a = Math.random() * Math.PI * 2, b = Math.random() * Math.PI - Math.PI / 2;
      fx.push({ obj: sp, life: 0, max: 30 + Math.random() * 25, kind: "p", grav,
        v: new T.Vector3(Math.cos(a) * Math.cos(b), Math.abs(Math.sin(b)) + 0.3, Math.sin(a) * Math.cos(b)).multiplyScalar(speed * (0.5 + Math.random())) });
    }
  }
  function orb(from, to, color, delay, dur, onArrive) {
    const sp = new T.Sprite(new T.SpriteMaterial({ map: glowTexture(), color, transparent: true, opacity: 0, depthWrite: false, blending: T.AdditiveBlending }));
    sp.scale.set(0.55, 0.55, 1); root.add(sp);
    const mid = from.clone().lerp(to, 0.5); mid.y += 1.4;
    tween(delay, dur, (t) => {
      const e = ease.inOut(t), a = from.clone().lerp(mid, e), b = mid.clone().lerp(to, e);
      sp.position.copy(a.lerp(b, e)); sp.material.opacity = Math.min(1, t * 4);
      if (Math.random() < 0.6) burst(sp.position, color, 1, 0.01, 0.12, 0);
    }, () => { root.remove(sp); sp.material.dispose(); onArrive && onArrive(); });
  }
  const headPos = (s) => new T.Vector3(s.position.x, 1.7, s.position.z + 0.1);
  const midPos = (s) => new T.Vector3(s.position.x, 0.8, s.position.z + 0.15);

  function flash(s, color, delay, dur = 360) {
    const u = s.userData, base = new T.Color(0xffffff), c = new T.Color(color);
    tween(delay, dur, (t) => { u.faceMat.color.copy(base).lerp(c, Math.sin(t * Math.PI) * 0.65); },
      () => u.faceMat.color.set(0xffffff));
  }
  function wobble(s, delay) {
    tween(delay, 380, (t) => { s.userData.lean.rotation.z = Math.sin(t * Math.PI * 3) * 0.12 * (1 - t); });
  }
  function glowPulse(s, delay, dur = 520) {
    const g = s.userData.glow;
    tween(delay, dur, (t) => { g.material.opacity = Math.sin(t * Math.PI) * 0.45; g.scale.setScalar(1.7 + Math.sin(t * Math.PI) * 0.5); });
  }
  function topple(s, delay, dur) {
    const u = s.userData;
    if (u.dead) return;
    u.dead = true;
    tween(delay, dur, (t) => {
      u.lean.rotation.x = -ease.out(t) * 1.25;
      u.faceMat.color.setRGB(1 - 0.5 * t, 1 - 0.5 * t, 1 - 0.5 * t);
      u.stats.material.opacity = 1 - t;
    });
  }
  function sinkAway(s, delay, dur) {
    burst(midPos(s), 0xb8c0bc, 16, 0.035, 0.5, 0.001);
    tween(delay, dur, (t) => { s.position.y = -ease.out(t) * 1.6; s.scale.setScalar(1 - t * 0.4); },
      () => disposeStandee(s));
  }

  // ------------------------------------------------------------- showing frames
  // Match last frame's pets to this frame's, per side, by name in order
  function matchSide(oldList, nowPets) {
    const used = new Set(), map = [];
    nowPets.forEach((p, k) => {
      let j = oldList.findIndex((s, i) => !used.has(i) && s.userData.p.name === p.name);
      if (j >= 0) used.add(j);
      map.push(j);
    });
    const gone = oldList.filter((_, i) => !used.has(i));
    return { map, gone };
  }

  function show(frame, prev, opts = {}) {
    if (!mounted) return;
    const animate = !!opts.animate && !!prev && !reduce.matches;
    const ms = Math.max(420, Math.min(opts.ms || 800, 1300));
    const next = [[], []];
    const changed = [new Set(), new Set()];
    for (const side of [0, 1]) {
      const nowPets = frame.teams[side];
      const { map, gone } = matchSide(pets[side], nowPets);
      nowPets.forEach((p, k) => {
        let s = map[k] >= 0 ? pets[side][map[k]] : null;
        const isNew = !s;
        if (isNew) {
          s = makeStandee(p, side);
          s.position.set(slotX(side, k), 0, 0);
          root.add(s);
          if (animate) {
            const u = s.userData;
            s.scale.setScalar(0.01);
            u.flip.rotation.y = Math.PI; u.stats.material.opacity = 0;
            tween(ms * 0.3, ms * 0.35, (t) => s.scale.setScalar(Math.max(0.01, ease.back(t))));
            tween(ms * 0.6, ms * 0.45, (t) => { u.flip.rotation.y = Math.PI * (1 - ease.inOut(t)); u.stats.material.opacity = t; },
              () => burst(midPos(s), 0xffe27a, 22, 0.06, 0.2));
            focus(slotX(side, k), 0.88, ms * 1.2);
          } else if (!prev) {
            // lineup: drop in from above
            s.position.y = 3 + k * 0.4;
            tween(k * 70 + side * 40, 520, (t) => { s.position.y = (3 + k * 0.4) * (1 - ease.out(t)); });
          }
        }
        const old = s.userData.p;
        const dh = p.health - old.health, da = p.attack - old.attack;
        if (!isNew && (dh || da || (old.status || {}).name !== (p.status || {}).name)) changed[side].add(k);
        s.userData.pending = { p, dh, da };
        next[side].push(s);
      });
      gone.forEach((s) => (animate ? sinkAway(s, ms * 0.1, ms * 0.6) : disposeStandee(s)));
    }
    pets = next;

    // slide everyone to their slot (pets move up after a faint)
    for (const side of [0, 1]) pets[side].forEach((s, k) => {
      const x0 = s.position.x, x1 = slotX(side, k);
      if (Math.abs(x0 - x1) < 0.01) return;
      if (!animate) { s.position.x = x1; return; }
      tween(ms * 0.15, ms * 0.5, (t) => { s.position.x = x0 + (x1 - x0) * ease.inOut(t); });
    });

    const events = frame.events || [];
    const impactAt = frame.phase === "phase_attack" ? ms * 0.38 : ms * 0.55;
    const find = (side, name) => pets[side].find((s) => s.userData.p.name.toLowerCase() === name);
    const reveal = (s, side, k, at) => {
      const { p, dh, da } = s.userData.pending;
      setTimeout(() => {
        setStats(s, p);
        if (dh < 0) { flash(s, 0xff4a3a, 0); wobble(s, 0); floatText(headPos(s), `${dh}`, css("--hp", "#c0392f")); burst(midPos(s), 0xff7a50, 10, 0.06, 0.16); }
        else if (dh > 0 || da > 0) {
          const f = (v) => (v > 0 ? "+" : "") + v;
          floatText(headPos(s), `${f(da)}/${f(dh)}`, css("--accent", "#1f6f5c")); burst(midPos(s), 0x7ce08a, 14, 0.04, 0.16, 0.001);
        }
        if (p.dead) topple(s, 120, ms * 0.5);
      }, at);
    };

    if (!animate) {
      for (const side of [0, 1]) pets[side].forEach((s) => {
        const { p } = s.userData.pending; setStats(s, p);
        s.userData.lean.rotation.x = p.dead ? -1.25 : 0; s.userData.dead = !!p.dead;
        s.userData.faceMat.color.set(p.dead ? 0x8c8c8c : 0xffffff);
        s.userData.stats.material.opacity = p.dead ? 0 : 1;
      });
      lastFrame = frame;
      return;
    }

    if (frame.phase === "phase_attack" && pets[0][0] && pets[1][0]) {
      // front pets lunge into each other
      const a = pets[0][0], b = pets[1][0];
      const ax = slotX(0, 0), bx = slotX(1, 0);
      focus(0, 0.8, ms * 0.85);
      tween(0, impactAt, (t) => { const e = ease.inOut(t); a.position.x = ax + (-0.5 - ax) * e; b.position.x = bx + (0.5 - bx) * e; a.position.y = b.position.y = Math.sin(t * Math.PI) * 0.35; },
        () => {
          shake = 0.16;
          burst(new T.Vector3(0, 0.9, 0.2), 0xffd36a, 26, 0.09, 0.2);
          tween(0, ms * 0.35, (t) => { const e = ease.out(t); a.position.x = -0.5 + (ax + 0.5) * e; b.position.x = 0.5 + (bx - 0.5) * e; });
        });
      for (const side of [0, 1]) changed[side].forEach((k) => reveal(pets[side][k], side, k, impactAt));
    } else {
      // abilities: the acting pet glows, an orb flies to each target
      const hit = [new Set(), new Set()];
      let delay = 0;
      const xs = [];
      events.forEach((ev) => {
        if (ev.kind === "Fainted" || ev.kind === "Attack") return;
        const src = find(ev.side, ev.src) || find(1 - ev.side, ev.src);
        if (src) xs.push(src.position.x);
        (ev.targets || []).forEach((name) => { const t = find(0, name) || find(1, name); if (t) xs.push(t.position.x); });
      });
      if (xs.length) {
        const lo = Math.min(...xs), hi = Math.max(...xs);
        focus((lo + hi) / 2, Math.min(1, Math.max(0.78, 0.7 + (hi - lo) / 14)), ms * 1.1);
      }
      events.forEach((ev) => {
        if (ev.kind === "Fainted" || ev.kind === "Attack") return;
        const src = find(ev.side, ev.src) || find(1 - ev.side, ev.src);
        if (src) glowPulse(src, delay);
        const color = /Damage/.test(ev.kind) ? 0xff6a4a : /Summon/.test(ev.kind) ? 0xffe27a : 0x7ce08a;
        (ev.targets || []).forEach((name, n) => {
          for (const side of [1 - ev.side, ev.side]) {
            const k = pets[side].findIndex((s, i) => s.userData.p.name.toLowerCase() === name && !hit[side].has(i) && (changed[side].has(i) || /Summon/.test(ev.kind)));
            if (k < 0) continue;
            hit[side].add(k);
            const tgt = pets[side][k];
            if (src && src !== tgt) orb(headPos(src), midPos(tgt), color, delay + 120 + n * 60, ms * 0.42);
            reveal(tgt, side, k, delay + (src && src !== tgt ? 120 + n * 60 + ms * 0.42 : 200));
            break;
          }
        });
        delay += ms * 0.18;
      });
      // anything that changed without a matching event (perks, chained effects)
      for (const side of [0, 1]) changed[side].forEach((k) => { if (!hit[side].has(k)) reveal(pets[side][k], side, k, impactAt); });
    }
    // pets that just took a fatal hit outside an attack/ability (already dead)
    for (const side of [0, 1]) pets[side].forEach((s, k) => {
      if (s.userData.pending.p.dead && !changed[side].has(k)) topple(s, ms * 0.3, ms * 0.5);
    });
    lastFrame = frame;
  }

  // ------------------------------------------------------------- render loop
  function frameCamera() {
    const w = el.clientWidth, h = el.clientHeight;
    camera.aspect = w / h;
    const need = 7.6;                              // half width that must fit
    const fov = 36 * Math.PI / 180;
    const dist = Math.max(8.2, need / (Math.tan(fov / 2) * camera.aspect));
    camBase = new T.Vector3(0, 2.4 + dist * 0.12, dist);
    camera.position.copy(camBase);
    camera.lookAt(0, 0.85, 0);
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }
  function loop(now) {
    raf = requestAnimationFrame(loop);
    for (let i = tweens.length - 1; i >= 0; i--) {
      const tw = tweens[i];
      if (now < tw.t0) continue;
      const t = Math.min(1, (now - tw.t0) / tw.dur);
      tw.fn(t);
      if (t >= 1) { tweens.splice(i, 1); tw.done && tw.done(); }
    }
    for (let i = fx.length - 1; i >= 0; i--) {
      const f = fx[i]; f.life++;
      const k = f.life / f.max;
      if (f.kind === "text") { f.obj.position.y += f.vy; f.obj.material.opacity = k > 0.6 ? (1 - k) / 0.4 : 1; }
      else { f.obj.position.add(f.v); f.v.y += f.grav; f.v.multiplyScalar(0.96); f.obj.material.opacity = 1 - k; }
      if (k >= 1) { root.remove(f.obj); if (f.kind === "text") f.obj.material.map.dispose(); f.obj.material.dispose(); fx.splice(i, 1); }
    }
    if (root.clouds && !reduce.matches) root.clouds.forEach((c) => { c.position.x += c.userData.v * 0.01; if (c.position.x > 30) c.position.x = -30; });
    // idle: standees sway a little so the field never looks frozen
    if (!reduce.matches) for (const side of [0, 1]) pets[side].forEach((s) => {
      const u = s.userData;
      if (!u.dead) u.lean.position.y = Math.sin(now / 700 + u.phase) * 0.025;
    });
    if (camBase) {
      if (now > cam.until) { cam.tx = 0; cam.tzoom = 1; }
      cam.x += (cam.tx - cam.x) * 0.07; cam.zoom += (cam.tzoom - cam.zoom) * 0.07;
      const sx = shake > 0.001 && !reduce.matches ? (Math.random() - 0.5) * shake : 0;
      const sy = shake > 0.001 && !reduce.matches ? (Math.random() - 0.5) * shake : 0;
      shake *= 0.86;
      camera.position.set(camBase.x + cam.x * 0.75 + sx, 0.9 + (camBase.y - 0.9) * cam.zoom + sy, camBase.z * cam.zoom);
      camera.lookAt(cam.x, 0.85, 0);
    }
    renderer.render(scene, camera);
  }

  async function mount(container) {
    await load();
    T = window.THREE;
    el = container;
    if (!renderer) {
      renderer = new T.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
      renderer.outputEncoding = T.sRGBEncoding;
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = T.PCFSoftShadowMap;
      camera = new T.PerspectiveCamera(36, 2, 0.1, 100);
    }
    if (!renderer.domElement.isConnected) el.appendChild(renderer.domElement);
    root = new T.Group();
    buildScene();
    pets = [[], []]; tweens = []; fx = [];
    Object.assign(cam, { x: 0, zoom: 1, tx: 0, tzoom: 1, until: 0 });
    frameCamera();
    ro = new ResizeObserver(frameCamera); ro.observe(el);
    mounted = true;
    cancelAnimationFrame(raf); raf = requestAnimationFrame(loop);
  }
  function unmount() {
    mounted = false;
    cancelAnimationFrame(raf); raf = 0;
    if (ro) ro.disconnect();
    if (scene) scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => { if (m.map && m.map !== glowTex && m.map !== backTex) m.map.dispose(); m.dispose(); }); });
    scene = null; pets = [[], []]; tweens = []; fx = []; lastFrame = null;
  }
  // Rebuild from scratch, e.g. when a new battle starts or the theme changes
  async function reset(container) { unmount(); await mount(container); }

  return { load, mount, unmount, reset, show, get mounted() { return mounted; } };
})();
