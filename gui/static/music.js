/*
 * Background music for sapai arena: two original tunes played live with Web
 * Audio, no audio files.
 *   "Meadow Market" (shop): 92 BPM, C major, soft plucks, pad and brushed beat
 *   "Scrapyard Rumble" (battle): 138 BPM, A minor, square lead, driving bass
 * Moods crossfade; music pauses while the tab is hidden.
 */
window.MUSIC = (() => {
  const N = (name) => {
    const m = /^([A-G])(#?)(\d)$/.exec(name);
    const base = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1]] + (m[2] ? 1 : 0);
    return 440 * Math.pow(2, (base + 12 * (Number(m[3]) + 1) - 69) / 12);
  };
  const chord = (...names) => names.map(N);

  const SONGS = {
    shop: {
      bpm: 92, swing: 0.08,
      chords: [chord("C3", "E3", "G3"), chord("A2", "C3", "E3"), chord("F2", "A2", "C3"), chord("G2", "B2", "D3"),
               chord("F2", "A2", "C3"), chord("G2", "B2", "D3"), chord("E2", "G2", "B2"), chord("A2", "C3", "E3")],
      melody: [
        [[0, "E5", 2], [2, "G5", 2], [4, "A5", 3], [8, "G5", 2], [10, "E5", 2], [12, "D5", 4]],
        [[0, "C5", 3], [4, "E5", 2], [6, "D5", 2], [8, "C5", 4], [12, "A4", 4]],
        [[0, "A4", 2], [2, "C5", 2], [4, "F5", 3], [8, "E5", 2], [10, "C5", 2], [12, "A4", 3]],
        [[0, "B4", 2], [2, "D5", 2], [4, "G5", 4], [10, "F5", 2], [12, "D5", 4]],
        [[0, "C6", 2], [2, "A5", 2], [4, "F5", 2], [6, "A5", 2], [8, "G5", 4], [12, "F5", 2], [14, "E5", 2]],
        [[0, "D5", 3], [4, "G5", 2], [6, "A5", 2], [8, "B5", 4], [12, "G5", 4]],
        [[0, "E5", 2], [2, "G5", 2], [4, "B5", 3], [8, "A5", 2], [10, "G5", 2], [12, "E5", 4]],
        [[0, "A5", 4], [4, "G5", 2], [6, "E5", 2], [8, "D5", 2], [10, "C5", 2], [12, "A4", 4]],
      ],
      bass: [0, 6, 8, 14],
      arp: [0, 2, 4, 6, 8, 10, 12, 14],
      kick: [0, 8], snare: [], hat: [2, 6, 10, 14], shaker: [1, 3, 5, 7, 9, 11, 13, 15],
      lead: { type: "triangle", gain: 0.16, release: 0.25 },
    },
    battle: {
      bpm: 138, swing: 0,
      chords: [chord("A2", "C3", "E3"), chord("F2", "A2", "C3"), chord("C3", "E3", "G3"), chord("G2", "B2", "D3"),
               chord("A2", "C3", "E3"), chord("F2", "A2", "C3"), chord("G2", "B2", "D3"), chord("E2", "G#2", "B2")],
      melody: [
        [[0, "A4", 2], [2, "C5", 2], [4, "E5", 2], [6, "A5", 2], [8, "G5", 3], [12, "E5", 2], [14, "C5", 2]],
        [[0, "F5", 3], [4, "E5", 1], [5, "F5", 1], [6, "A5", 2], [8, "C6", 4], [12, "A5", 4]],
        [[0, "G5", 2], [2, "E5", 2], [4, "C5", 2], [6, "E5", 2], [8, "G5", 2], [10, "C6", 2], [12, "B5", 2], [14, "G5", 2]],
        [[0, "D5", 2], [2, "G5", 2], [4, "B5", 4], [8, "A5", 2], [10, "G5", 2], [12, "D5", 4]],
        [[0, "A4", 2], [2, "C5", 2], [4, "E5", 2], [6, "A5", 2], [8, "G5", 3], [12, "A5", 4]],
        [[0, "C6", 2], [2, "A5", 2], [4, "F5", 2], [6, "A5", 2], [8, "C6", 2], [10, "D6", 2], [12, "C6", 4]],
        [[0, "B5", 2], [2, "D6", 2], [4, "B5", 2], [6, "G5", 2], [8, "D5", 4], [12, "G5", 4]],
        [[0, "E5", 2], [2, "G#5", 2], [4, "B5", 2], [6, "E6", 2], [8, "D6", 2], [10, "B5", 2], [12, "G#5", 4]],
      ],
      bass: [0, 2, 4, 6, 8, 10, 12, 14], bassOct: [0, 0, 1, 0, 0, 0, 1, 0],
      arp: [],
      kick: [0, 4, 8, 12], snare: [4, 12], hat: [0, 2, 4, 6, 8, 10, 12, 14], shaker: [],
      lead: { type: "square", gain: 0.07, release: 0.12 },
    },
  };

  let ctx = null, bus = null, comp = null;
  let enabled = true, volume = 0.35, mood = "shop";
  let playing = null, timer = null, step = 0, nextTime = 0, songGain = null;
  try {
    const v = JSON.parse(localStorage.getItem("sapaiMusic") || "{}");
    if (typeof v.enabled === "boolean") enabled = v.enabled;
    if (typeof v.volume === "number") volume = v.volume;
  } catch (e) {}
  const persist = () => { try { localStorage.setItem("sapaiMusic", JSON.stringify({ enabled, volume })); } catch (e) {} };

  function init() {
    if (ctx) return ctx;
    const C = window.AudioContext || window.webkitAudioContext;
    if (!C) return null;
    ctx = new C();
    comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -18; comp.ratio.value = 4;
    bus = ctx.createGain();
    bus.gain.value = volume * 0.6;
    bus.connect(comp); comp.connect(ctx.destination);
    return ctx;
  }

  function note(out, freq, t, len, type, gain, attack = 0.01, release = 0.2, filter = 0) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, t);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + attack);
    g.gain.setValueAtTime(gain, t + Math.max(attack, len - release * 0.5));
    g.gain.exponentialRampToValueAtTime(0.0001, t + len + release);
    let node = o;
    if (filter) { const f = ctx.createBiquadFilter(); f.type = "lowpass"; f.frequency.value = filter; o.connect(f); node = f; }
    node.connect(g); g.connect(out);
    o.start(t); o.stop(t + len + release + 0.05);
  }
  let noiseBuf = null;
  function hitNoise(out, t, len, freq, gain, type = "highpass") {
    if (!noiseBuf) {
      noiseBuf = ctx.createBuffer(1, ctx.sampleRate * 0.5, ctx.sampleRate);
      const d = noiseBuf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    const s = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
    s.buffer = noiseBuf; f.type = type; f.frequency.value = freq;
    g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + len);
    s.connect(f); f.connect(g); g.connect(out);
    s.start(t); s.stop(t + len + 0.02);
  }
  function kick(out, t, gain) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.setValueAtTime(140, t); o.frequency.exponentialRampToValueAtTime(45, t + 0.18);
    g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.25);
    o.connect(g); g.connect(out); o.start(t); o.stop(t + 0.3);
  }

  function schedule() {
    const song = SONGS[playing];
    const sixteenth = 60 / song.bpm / 4;
    while (nextTime < ctx.currentTime + 0.25) {
      const bar = Math.floor(step / 16) % song.chords.length, s = step % 16;
      const t = nextTime + (s % 2 ? song.swing * sixteenth : 0);
      const ch = song.chords[bar], out = songGain;
      // pad: whole bar chord, very soft
      if (s === 0) for (const f of ch) note(out, f * 2, t, sixteenth * 15, "sawtooth", playing === "shop" ? 0.018 : 0.014, 0.3, 0.5, 900);
      // bass
      const bi = song.bass.indexOf(s);
      if (bi >= 0) {
        const oct = song.bassOct ? song.bassOct[bi] : 0;
        note(out, ch[0] * (oct ? 2 : 1), t, sixteenth * (playing === "shop" ? 3 : 1.6), playing === "shop" ? "sine" : "triangle", playing === "shop" ? 0.22 : 0.2, 0.01, 0.08);
      }
      // arpeggio plucks (shop)
      if (song.arp.includes(s)) note(out, ch[(s / 2) % 3] * 4, t, sixteenth * 1.2, "triangle", 0.035, 0.005, 0.15);
      // melody
      for (const [ms, name, len] of song.melody[bar]) {
        if (ms === s) note(out, N(name), t, sixteenth * len * 0.92, song.lead.type, song.lead.gain, 0.01, song.lead.release, playing === "battle" ? 2600 : 0);
      }
      // drums
      if (song.kick.includes(s)) kick(out, t, playing === "shop" ? 0.28 : 0.42);
      if (song.snare.includes(s)) { hitNoise(out, t, 0.14, 1800, 0.22, "bandpass"); note(out, 190, t, 0.05, "triangle", 0.12, 0.002, 0.06); }
      if (song.hat.includes(s)) hitNoise(out, t, playing === "shop" ? 0.04 : 0.03, 7000, playing === "shop" ? 0.05 : 0.07);
      if (song.shaker.includes(s)) hitNoise(out, t, 0.025, 9000, 0.02);
      step++;
      nextTime += sixteenth;
    }
  }

  function startSong(name) {
    if (!init()) return;
    if (ctx.state === "suspended") ctx.resume();
    const now = ctx.currentTime;
    if (songGain) {
      const old = songGain;
      old.gain.cancelScheduledValues(now);
      old.gain.setValueAtTime(old.gain.value, now);
      old.gain.linearRampToValueAtTime(0.0001, now + 0.6);
      setTimeout(() => old.disconnect(), 900);
    }
    songGain = ctx.createGain();
    songGain.gain.setValueAtTime(0.0001, now);
    songGain.gain.linearRampToValueAtTime(1, now + 0.6);
    songGain.connect(bus);
    playing = name; step = 0; nextTime = now + 0.05;
    clearInterval(timer);
    timer = setInterval(schedule, 60);
    schedule();
  }
  function stop() {
    clearInterval(timer); timer = null;
    if (songGain && ctx) {
      const g = songGain, now = ctx.currentTime;
      g.gain.cancelScheduledValues(now); g.gain.setValueAtTime(g.gain.value, now);
      g.gain.linearRampToValueAtTime(0.0001, now + 0.4);
      setTimeout(() => g.disconnect(), 600);
    }
    songGain = null; playing = null;
  }
  function apply() {
    if (!enabled || volume === 0 || document.hidden) { stop(); return; }
    if (playing !== mood) startSong(mood);
    if (bus) bus.gain.value = volume * 0.6;
  }
  document.addEventListener("visibilitychange", apply);
  // Browsers only allow audio after the first click or key press
  let unlocked = false;
  const unlock = () => { if (unlocked) return; unlocked = true; apply(); };
  addEventListener("pointerdown", unlock, { once: false });
  addEventListener("keydown", unlock, { once: false });

  return {
    setMood(m) { mood = m; if (unlocked) apply(); },
    get enabled() { return enabled; }, get volume() { return volume; },
    setEnabled(v) { enabled = v; persist(); unlocked = true; apply(); },
    setVolume(v) { volume = v; if (v > 0) enabled = true; persist(); unlocked = true; apply(); },
  };
})();
