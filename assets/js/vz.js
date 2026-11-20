/* ============================================================
   Vectorization Guide — interactive labs (vanilla JS, no dependencies)
   A page opts in with  <div class="lab" data-lab="NAME"></div>
   NAME ∈ lanes | deps | stride | mask | roofline | reduce | systolic | broadcast | amdahl | atlas
   Every lab mirrors a function in code/vzlab so the numbers on the page and in the tests agree.
   ============================================================ */
(function () {
  'use strict';

  var LANE = ['#22d3ee', '#a78bfa', '#34d399', '#fbbf24', '#f472b6', '#60a5fa', '#fb923c', '#a3e635'];
  var C = { bg: '#0b0b0b', grid: 'rgba(255,255,255,0.08)', text: '#e8e8e8', dim: '#666', mid: '#999',
            acc: '#22d3ee', red: '#ef4444', green: '#22c55e', pink: '#f472b6', yellow: '#fbbf24', purple: '#a78bfa', blue: '#60a5fa' };

  // ------------------------------------------------------------ helpers
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function canvas(w, h) {
    var c = el('canvas'), r = window.devicePixelRatio || 1;
    c.width = w * r; c.height = h * r; c.style.width = w + 'px'; c.style.height = h + 'px';
    c.getContext('2d').setTransform(r, 0, 0, r, 0, 0);
    c._w = w; c._h = h;
    return c;
  }
  function frame(root, title) {
    root.innerHTML = '';
    root.appendChild(el('div', { 'class': 'lab-title' }, title));
    var body = el('div', { 'class': 'lab-body' });
    root.appendChild(body);
    return body;
  }
  function panel(body) { var p = el('div', { 'class': 'lab-panel' }); body.appendChild(p); return p; }
  function controls(parent) { var c = el('div', { 'class': 'lab-controls' }); parent.appendChild(c); return c; }
  function select(parent, label, opts, val) {
    var l = el('label', null, label), s = el('select');
    opts.forEach(function (o) {
      var v = Array.isArray(o) ? o[0] : o, t = Array.isArray(o) ? o[1] : o;
      var op = el('option', { value: v }, t); if (String(v) === String(val)) op.selected = true; s.appendChild(op);
    });
    l.appendChild(s); parent.appendChild(l); return s;
  }
  function slider(parent, label, min, max, step, val) {
    var l = el('label', null, label), s = el('input', { type: 'range', min: min, max: max, step: step, value: val });
    var out = el('span', { style: 'font-family:var(--font-mono);min-width:3.2em;display:inline-block' }, String(val));
    s.addEventListener('input', function () { out.textContent = s.value; });
    l.appendChild(s); l.appendChild(out); parent.appendChild(l); s._out = out; return s;
  }
  function button(parent, label) { var b = el('button', null, label); parent.appendChild(b); return b; }
  function readout(parent) { var r = el('div', { 'class': 'lab-readout' }); parent.appendChild(r); return r; }
  function hint(parent, txt) { parent.appendChild(el('div', { 'class': 'lab-hint' }, txt)); }
  function fmt(x, d) { return (x === null || x === undefined || isNaN(x)) ? '—' : Number(x).toFixed(d === undefined ? 2 : d); }
  function txt(ctx, s, x, y, color, size, align, font) {
    ctx.fillStyle = color || C.text; ctx.font = (size || 11) + 'px ' + (font || 'JetBrains Mono, monospace');
    ctx.textAlign = align || 'left'; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y);
  }
  function rect(ctx, x, y, w, h, fill, stroke) {
    if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
  }
  function clear(ctx, c) { ctx.fillStyle = C.bg; ctx.fillRect(0, 0, c._w, c._h); }
  function arrow(ctx, x1, y1, x2, y2, color, dashed, bend) {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash(dashed ? [4, 3] : []);
    var mx = (x1 + x2) / 2, my = Math.min(y1, y2) - (bend || 0);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(mx, my, x2, y2); ctx.stroke(); ctx.setLineDash([]);
    var a = Math.atan2(y2 - my, x2 - mx);
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - 7 * Math.cos(a - 0.4), y2 - 7 * Math.sin(a - 0.4));
    ctx.lineTo(x2 - 7 * Math.cos(a + 0.4), y2 - 7 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
  }
  function rng(seed) { var s = seed >>> 0 || 1; return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

  // ------------------------------------------------------------ 1. lanes: scalar vs SIMD, tail strategies (vzlab.perfmodel.tail_cost)
  function labLanes(root) {
    var body = frame(root, 'Lab · Scalar vs SIMD — who processes which element');
    var c = canvas(560, 250); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var nS = slider(ct, 'n', 1, 40, 1, 27), vS = select(ct, 'lanes V', [1, 4, 8, 16], 8);
    var tS = select(ct, 'tail', [['epilogue', 'scalar epilogue'], ['masked', 'masked last vector'], ['pad', 'pad to multiple of V']], 'epilogue');
    var ct2 = controls(p), stepB = button(ct2, 'Step'), playB = button(ct2, 'Play'), resetB = button(ct2, 'Reset');
    var out = readout(p);
    hint(p, 'Each instruction covers V elements at once. Try n = 27 with V = 16: the tail is most of the work.');
    var plan = [], done = 0, timer = null;
    function build() {
      var n = +nS.value, V = +vS.value, t = tS.value; plan = [];
      var full = Math.floor(n / V), rem = n % V, i;
      for (i = 0; i < full; i++) plan.push({ start: i * V, len: V, kind: 'vec' });
      if (rem) {
        if (t === 'epilogue') for (i = 0; i < rem; i++) plan.push({ start: full * V + i, len: 1, kind: 'scalar' });
        else plan.push({ start: full * V, len: V, active: rem, kind: t === 'masked' ? 'mask' : 'pad' });
      }
      done = 0; draw();
    }
    function draw() {
      clear(ctx, c);
      var n = +nS.value, V = +vS.value, cw = Math.min(26, 520 / Math.max(n, 1)), x0 = 20, y0 = 40;
      var padTo = tS.value === 'pad' ? Math.ceil(n / V) * V : n;
      txt(ctx, 'array a[0..' + (n - 1) + ']', x0, 18, C.mid, 11);
      for (var i = 0; i < Math.max(n, padTo); i++) {
        var owner = -1;
        for (var k = 0; k < done; k++) { var s = plan[k]; if (i >= s.start && i < s.start + s.len) owner = k; }
        var fill = '#1e1e1e', isPad = i >= n;
        if (owner >= 0) {
          var st = plan[owner];
          fill = st.kind === 'scalar' ? '#888' : LANE[(i - st.start) % 8];
          if ((st.kind === 'mask' || st.kind === 'pad') && i >= n) fill = st.kind === 'mask' ? '#2a2a2a' : '#5b3a3a';
        }
        rect(ctx, x0 + i * cw, y0, cw - 2, 28, fill, isPad ? '#7f3b3b' : '#333');
        if (cw > 15) txt(ctx, String(i), x0 + i * cw + (cw - 2) / 2, y0 + 14, owner >= 0 ? '#000' : C.dim, 9, 'center');
      }
      // instruction log
      var y = y0 + 52; txt(ctx, 'instruction stream', x0, y, C.mid, 11); y += 16;
      var show = plan.slice(Math.max(0, done - 7), done);
      show.forEach(function (s, j) {
        var label = s.kind === 'vec' ? 'vfmadd  a[' + s.start + '..' + (s.start + s.len - 1) + ']   ' + s.len + ' lanes' :
          s.kind === 'scalar' ? 'fmadd   a[' + s.start + ']            1 lane' :
          s.kind === 'mask' ? 'vfmadd{k} a[' + s.start + '..' + (s.start + s.len - 1) + ']  mask ' + s.active + '/' + s.len :
          'vfmadd  a[' + s.start + '..' + (s.start + s.len - 1) + ']   ' + (s.len - s.active) + ' pad lanes';
        txt(ctx, (Math.max(0, done - 7) + j + 1) + '. ' + label, x0, y + j * 16, s.kind === 'scalar' ? '#bbb' : C.acc, 11);
      });
      var lanes = plan.reduce(function (a, s) { return a + (s.kind === 'scalar' ? 1 : s.len); }, 0);
      out.textContent = 'elements n          ' + n + '\nlanes V             ' + V +
        '\ninstructions        ' + plan.length + '   (scalar loop: ' + n + ')' +
        '\nspeedup (instr.)    ' + fmt(n / plan.length, 2) + '×  of ideal ' + V + '×' +
        '\nlane utilisation    ' + fmt(100 * n / lanes, 1) + ' %' +
        '\nissued so far       ' + done + ' / ' + plan.length;
    }
    function step() { if (done < plan.length) { done++; draw(); } else if (timer) { clearInterval(timer); timer = null; } }
    stepB.onclick = step;
    playB.onclick = function () { if (timer) { clearInterval(timer); timer = null; return; } if (done >= plan.length) done = 0; timer = setInterval(step, 350); };
    resetB.onclick = function () { if (timer) { clearInterval(timer); timer = null; } done = 0; draw(); };
    [nS, vS, tS].forEach(function (x) { x.addEventListener('input', build); x.addEventListener('change', build); });
    build(); done = plan.length; draw();
  }

  // ------------------------------------------------------------ 2. deps: dependence distance & legality (vzlab.deps)
  function labDeps(root) {
    var body = frame(root, 'Lab · Dependence distance — may these iterations run in the same vector?');
    var c = canvas(560, 230); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var rS = slider(ct, 'read  a[i + R]  R', -6, 6, 1, 0), wS = slider(ct, 'write a[i + W]  W', -6, 18, 1, 1);
    var vS = select(controls(p), 'lanes V', [2, 4, 8, 16], 8);
    var out = readout(p);
    hint(p, 'Loop:  for i: a[i+W] = 0.5*a[i+R] + b[i].  Red arrows cross inside one vector chunk: that is what breaks.');
    function simulate(R, W, V) {
      var n = 48, off = Math.max(0, -Math.min(R, W)), size = n + off + Math.max(R, W, 0) + 1, a = [], b = [], i, k;
      for (k = 0; k < size; k++) a.push((k % 7) - 3);
      for (k = 0; k < n; k++) b.push(0.25 * (k % 5));
      var s = a.slice(), v = a.slice();
      for (i = 0; i < n; i++) s[i + W + off] = 0.5 * s[i + R + off] + b[i];
      for (var base = 0; base < n; base += V) {
        var idx = []; for (i = base; i < Math.min(base + V, n); i++) idx.push(i);
        var loads = idx.map(function (j) { return v[j + R + off]; });
        idx.forEach(function (j, q) { v[j + W + off] = 0.5 * loads[q] + b[j]; });
      }
      for (k = 0; k < size; k++) if (s[k] !== v[k]) return false;
      return true;
    }
    function draw() {
      var R = +rS.value, W = +wS.value, V = +vS.value, d = W - R; clear(ctx, c);
      var N = 24, cw = 22, x0 = 16, y0 = 130;
      for (var base = 0; base < N; base += V) rect(ctx, x0 + base * cw - 2, y0 - 22, Math.min(V, N - base) * cw, 58, (base / V) % 2 ? 'rgba(34,211,238,0.06)' : 'rgba(167,139,250,0.08)');
      for (var i = 0; i < N; i++) {
        rect(ctx, x0 + i * cw, y0, cw - 3, 22, '#1d2a2e', '#2f4a50'); txt(ctx, String(i), x0 + i * cw + 9.5, y0 + 11, C.text, 9, 'center');
      }
      txt(ctx, 'iterations i (shaded blocks = one vector instruction of V lanes)', x0, y0 + 50, C.mid, 10);
      if (d !== 0) for (i = 0; i < N; i++) {
        var j = i + Math.abs(d); if (j >= N) continue;
        var flow = d > 0, sameChunk = Math.floor(i / V) === Math.floor(j / V);
        var col = flow ? (sameChunk ? C.red : C.green) : C.blue;
        arrow(ctx, x0 + i * cw + 10, y0 - 2, x0 + j * cw + 10, y0 - 2, col, !flow, 18 + Math.min(80, Math.abs(d) * 6));
      }
      var legalA = d <= 0 || d >= V, legalS = simulate(R, W, V);
      var kind = d > 0 ? 'flow (read-after-write): iteration i+' + d + ' reads what i wrote' : d < 0 ? 'anti (write-after-read): later iteration overwrites what i read' : 'same element, same iteration';
      out.innerHTML = 'distance d = W − R = ' + d + '\n' + kind +
        '\nmax safe V          ' + (d <= 0 ? 'unlimited' : d) +
        '\nrule  d ≤ 0 or d ≥ V  ' + (legalA ? '<span class="chip ok">legal</span>' : '<span class="chip no">illegal</span>') +
        '\nsimulation agrees?  ' + (legalS === legalA ? 'yes' : 'NO') + ' (scalar vs chunked run on 48 elements)';
    }
    [rS, wS, vS].forEach(function (x) { x.addEventListener('input', draw); x.addEventListener('change', draw); });
    draw();
  }

  // ------------------------------------------------------------ 3. stride: cache lines & coalescing (vzlab.memory)
  function labStride(root) {
    var body = frame(root, 'Lab · Stride, layout and the bytes you actually move');
    var c = canvas(560, 270); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var mS = select(ct, 'target', [['cpu', 'CPU: 64-byte cache lines, 16 accesses'], ['gpu', 'GPU warp: 32-byte sectors, 32 threads']], 'cpu');
    var lS = select(ct, 'pattern', [['stride', 'strided a[i*s]'], ['aos', 'AoS: read x,y,z of 8-field struct'], ['soa', 'SoA: read x only']], 'stride');
    var sS = slider(controls(p), 'stride s', 1, 32, 1, 1), eS = select(controls(p), 'element bytes', [1, 2, 4, 8], 4);
    var out = readout(p);
    hint(p, 'A cache line is the unit of transfer. Stride 16 on fp32 uses 4 of every 64 bytes it moves.');
    function draw() {
      clear(ctx, c);
      var gpu = mS.value === 'gpu', line = gpu ? 32 : 64, nAcc = gpu ? 32 : 16, eb = +eS.value, s = +sS.value, pat = lS.value;
      var addrs = [], i;
      if (pat === 'stride') for (i = 0; i < nAcc; i++) addrs.push(i * s * eb);
      else if (pat === 'aos') for (i = 0; i < Math.ceil(nAcc / 3); i++) for (var f = 0; f < 3 && addrs.length < nAcc; f++) addrs.push((i * 8 + f) * eb);
      else for (i = 0; i < nAcc; i++) addrs.push(i * eb);
      var maxA = Math.max.apply(null, addrs) + eb, nLines = Math.ceil(maxA / line), per = line / eb;
      var touched = {}; addrs.forEach(function (a) { touched[Math.floor(a / line)] = 1; });
      var cell = Math.max(3, Math.min(14, Math.floor(520 / Math.min(per * 2, 128)))), rowsMax = 12;
      var linesPerRow = Math.max(1, Math.floor(520 / (per * cell + 6)));
      txt(ctx, (gpu ? 'sectors' : 'cache lines') + ' (each box = one element; ' + line + ' B per ' + (gpu ? 'sector' : 'line') + ')', 16, 14, C.mid, 10);
      var shown = Math.min(nLines, rowsMax * linesPerRow);
      for (var L = 0; L < shown; L++) {
        var rx = 16 + (L % linesPerRow) * (per * cell + 6), ry = 28 + Math.floor(L / linesPerRow) * 20;
        rect(ctx, rx - 2, ry - 2, per * cell + 3, 18, null, touched[L] ? C.acc : '#333');
        for (var e = 0; e < per; e++) {
          var a = L * line + e * eb, used = addrs.indexOf(a) >= 0;
          rect(ctx, rx + e * cell, ry, cell - 1, 14, used ? (pat === 'aos' ? LANE[[4, 3, 2][addrs.indexOf(a) % 3]] : C.acc) : (touched[L] ? '#4a585c' : '#1b1b1b'));
        }
      }
      if (nLines > shown) txt(ctx, '… ' + (nLines - shown) + ' more ' + (gpu ? 'sectors' : 'lines'), 16, 28 + rowsMax * 20 + 6, C.mid, 10);
      var moved = Object.keys(touched).length * line, useful = nAcc * eb;
      out.textContent = (gpu ? 'threads in warp      ' : 'accesses             ') + nAcc +
        '\n' + (gpu ? 'sectors touched      ' : 'cache lines touched  ') + Object.keys(touched).length +
        '\nbytes moved          ' + moved + '\nbytes used           ' + useful +
        '\nefficiency           ' + fmt(100 * useful / moved, 1) + ' %' +
        (gpu ? '\n(fully coalesced fp32 warp = 4 sectors = 128 B)' : '');
      sS.disabled = pat !== 'stride';
    }
    [mS, lS, sS, eS].forEach(function (x) { x.addEventListener('input', draw); x.addEventListener('change', draw); });
    draw();
  }

  // ------------------------------------------------------------ 4. mask: if/else under SIMD (vzlab.perfmodel.branch_utilization)
  function labMask(root) {
    var body = frame(root, 'Lab · Masks and divergence — both sides of the branch, every time');
    var c = canvas(560, 260); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var vS = select(ct, 'lanes V', [4, 8, 16, 32], 16), pS = slider(ct, 'P(condition)', 0, 1, 0.05, 0.5);
    var gS = select(controls(p), 'data order', [['random', 'random'], ['sorted', 'sorted by condition'], ['alt', 'alternating'], ['clustered', 'clustered runs of 8']], 'random');
    var reB = button(controls(p), 'Re-roll');
    var out = readout(p);
    hint(p, 'if (x > t) A else B. A chunk pays for side A if any lane wants A, and for B if any lane wants B. Sort the data and watch utilisation jump.');
    var seed = 7;
    function cond() {
      var n = 128, r = rng(seed), P = +pS.value, out = [], i;
      for (i = 0; i < n; i++) out.push(r() < P);
      if (gS.value === 'sorted') out.sort(function (a, b) { return b - a; });
      if (gS.value === 'alt') for (i = 0; i < n; i++) out[i] = i % 2 === 0;
      if (gS.value === 'clustered') { out = []; for (i = 0; i < n; i += 8) { var v = r() < P; for (var j = 0; j < 8; j++) out.push(v); } }
      return out;
    }
    function draw() {
      clear(ctx, c);
      var V = +vS.value, cs = cond(), chunks = Math.ceil(cs.length / V), cell = Math.min(15, Math.floor(400 / V));
      var useful = 0, issued = 0, passes = 0, rows = Math.min(chunks, 12);
      txt(ctx, 'chunk', 10, 12, C.mid, 10); txt(ctx, 'side A pass', 70 + V * cell + 10, 12, C.acc, 10); txt(ctx, 'side B pass', 70 + V * cell + 90, 12, C.pink, 10);
      for (var k = 0; k < chunks; k++) {
        var ch = cs.slice(k * V, k * V + V), t = ch.filter(Boolean).length, f = ch.length - t;
        if (t) { issued += V; useful += t; passes++; }
        if (f) { issued += V; useful += f; passes++; }
        if (k >= rows) continue;
        var y = 24 + k * 19; txt(ctx, String(k), 14, y + 7, C.mid, 10);
        ch.forEach(function (b, i) { rect(ctx, 40 + i * cell, y, cell - 2, 14, b ? C.acc : C.pink); });
        if (t) rect(ctx, 70 + V * cell + 10, y, 60 * t / V, 14, C.acc, '#555'); else txt(ctx, 'skipped', 70 + V * cell + 12, y + 7, C.dim, 9);
        if (f) rect(ctx, 70 + V * cell + 90, y, 60 * f / V, 14, C.pink, '#555'); else txt(ctx, 'skipped', 70 + V * cell + 92, y + 7, C.dim, 9);
      }
      if (chunks > rows) txt(ctx, '… ' + (chunks - rows) + ' more chunks', 40, 24 + rows * 19 + 6, C.mid, 10);
      out.textContent = 'elements             ' + cs.length + '\nchunks of V          ' + chunks +
        '\npasses issued        ' + passes + '   (no-branch ideal: ' + chunks + ')' +
        '\nlane utilisation     ' + fmt(100 * useful / issued, 1) + ' %' +
        '\nchunks needing both  ' + fmt(100 * (passes - chunks) / chunks, 0) + ' %';
    }
    [vS, pS, gS].forEach(function (x) { x.addEventListener('input', draw); x.addEventListener('change', draw); });
    reB.onclick = function () { seed = (seed * 31 + 11) % 100000; draw(); };
    draw();
  }

  // ------------------------------------------------------------ 5. roofline with vector roofs (vzlab.perfmodel.roofline)
  // Default ceilings are the measured single-core numbers in tools/measurements/results.json
  // (Xeon, Emerald Rapids, gcc 13.3): compute roofs from register-only FMA chains, bandwidth from saxpy.
  var ROOF = { roofs: [['scalar', 12.9], ['SSE 128-bit', 51.7], ['AVX2 256-bit', 103.1], ['AVX-512', 185.8]],
               bw: [['L1', 314], ['L2', 137], ['L3', 39], ['DRAM', 21]] };
  function labRoofline(root) {
    var body = frame(root, 'Lab · Roofline — when does a wider vector help?');
    var c = canvas(560, 320); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var presets = [['0.167', 'saxpy y=ax+y (2 FLOP / 12 B)'], ['0.25', 'dot product (2 FLOP / 8 B)'], ['0.125', 'leaky ReLU (1 FLOP / 8 B)'],
      ['0.6', 'softmax row, fused (~5 FLOP / 8 B)'], ['2.7', 'matmul 512, ikj (row of B from L2)'], ['42', 'blocked GEMM tile 256'], ['custom', 'custom…']];
    var kS = select(ct, 'kernel', presets, '0.167'), mS = select(ct, 'data lives in', ROOF.bw.map(function (b) { return [b[0], b[0] + ' (' + b[1] + ' GB/s)']; }), 'DRAM');
    var iS = slider(controls(p), 'intensity log₁₀(FLOP/B)', -2, 2.5, 0.01, Math.log10(0.167).toFixed(2));
    var out = readout(p);
    hint(p, 'Below the ridge the kernel is limited by the slanted bandwidth line: a wider register raises only the flat roofs.');
    kS.addEventListener('change', function () { if (kS.value !== 'custom') { iS.value = Math.log10(+kS.value).toFixed(2); iS._out.textContent = iS.value; } draw(); });
    iS.addEventListener('input', function () { kS.value = 'custom'; draw(); });
    mS.addEventListener('change', draw);
    function draw() {
      clear(ctx, c);
      var L = 50, R = 545, T = 15, B = 285, xmin = -2, xmax = 2.5, ymin = -0.5, ymax = 2.6;
      function X(lg) { return L + (lg - xmin) / (xmax - xmin) * (R - L); }
      function Y(lg) { return B - (lg - ymin) / (ymax - ymin) * (B - T); }
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      for (var g = -2; g <= 2; g++) { ctx.beginPath(); ctx.moveTo(X(g), T); ctx.lineTo(X(g), B); ctx.stroke(); txt(ctx, Math.pow(10, g) + '', X(g), B + 12, C.mid, 9, 'center'); }
      for (g = 0; g <= 2; g++) { ctx.beginPath(); ctx.moveTo(L, Y(g)); ctx.lineTo(R, Y(g)); ctx.stroke(); txt(ctx, Math.pow(10, g) + '', L - 6, Y(g), C.mid, 9, 'right'); }
      txt(ctx, 'arithmetic intensity (FLOP / byte)', (L + R) / 2, B + 26, C.mid, 10, 'center');
      txt(ctx, 'GFLOP/s', 8, T + 4, C.mid, 9);
      var bwName = mS.value, bw = ROOF.bw.filter(function (b) { return b[0] === bwName; })[0][1];
      ROOF.bw.forEach(function (b) {
        ctx.strokeStyle = b[0] === bwName ? '#ffffff' : 'rgba(255,255,255,0.18)'; ctx.lineWidth = b[0] === bwName ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(X(xmin), Y(Math.log10(b[1]) + xmin)); ctx.lineTo(X(xmax), Y(Math.log10(b[1]) + xmax)); ctx.stroke();
        var lx = Math.log10(3 / b[1]) + 0.05; if (lx > xmin) txt(ctx, b[0], X(lx), Y(Math.log10(3)) - 8, b[0] === bwName ? '#fff' : C.dim, 9);
      });
      ROOF.roofs.forEach(function (r, k) {
        ctx.strokeStyle = LANE[[2, 5, 1, 0][k]]; ctx.lineWidth = 2;
        var ridge = Math.log10(r[1] / bw);
        ctx.beginPath(); ctx.moveTo(X(Math.max(xmin, ridge)), Y(Math.log10(r[1]))); ctx.lineTo(X(xmax), Y(Math.log10(r[1]))); ctx.stroke();
        txt(ctx, r[0] + ' ' + r[1], X(xmax) - 4, Y(Math.log10(r[1])) - 8, LANE[[2, 5, 1, 0][k]], 9, 'right');
      });
      var I = Math.pow(10, +iS.value);
      ctx.strokeStyle = C.yellow; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(+iS.value), T); ctx.lineTo(X(+iS.value), B); ctx.stroke(); ctx.setLineDash([]);
      var lines = ['intensity I        ' + fmt(I, 3) + ' FLOP/B', 'bandwidth (' + bwName + ')    ' + bw + ' GB/s', ''];
      ROOF.roofs.forEach(function (r, k) {
        var att = Math.min(r[1], I * bw);
        ctx.fillStyle = LANE[[2, 5, 1, 0][k]]; ctx.beginPath(); ctx.arc(X(+iS.value), Y(Math.log10(att)), 4, 0, 7); ctx.fill();
        lines.push((r[0] + '              ').slice(0, 14) + fmt(att, 1) + ' GFLOP/s  ' + (att < r[1] ? '(memory-bound)' : '(compute-bound)'));
      });
      var s0 = Math.min(ROOF.roofs[0][1], I * bw), s3 = Math.min(ROOF.roofs[3][1], I * bw);
      lines.push('', 'AVX-512 over scalar  ' + fmt(s3 / s0, 2) + '×  (ceiling, not a promise)');
      out.textContent = lines.join('\n');
    }
    draw();
  }

  // ------------------------------------------------------------ 6. reduce: summation order in float32 (vzlab.reduce)
  function labReduce(root) {
    var body = frame(root, 'Lab · Same numbers, different order, different answer');
    var c = canvas(560, 230); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var nS = select(ct, 'n', [[1000, '1 000'], [10000, '10 000'], [100000, '100 000'], [1000000, '1 000 000']], 100000);
    var dS = select(ct, 'data', [['big', 'N(10⁴, 10³²)'], ['unit', 'uniform [0, 1)'], ['mixed', 'alternating ±10⁸ and small']], 'big');
    var reB = button(controls(p), 'Re-roll');
    var out = readout(p);
    hint(p, 'Each bar is |error| in units-in-the-last-place of the exact float32 answer. A SIMD loop with V accumulators is a different order from the C source.');
    var f = Math.fround, seed = 3;
    function data() {
      var n = +nS.value, r = rng(seed), x = new Float32Array(n), i;
      for (i = 0; i < n; i++) {
        if (dS.value === 'big') { var u = r(), v = r(); x[i] = 1e4 + 1e3 * Math.sqrt(-2 * Math.log(u + 1e-12)) * Math.cos(6.283185307 * v); }
        else if (dS.value === 'unit') x[i] = r();
        else x[i] = i % 2 ? (i % 4 === 1 ? 1e8 : -1e8) : r();
      }
      return x;
    }
    function seq(x) { var s = 0; for (var i = 0; i < x.length; i++) s = f(s + x[i]); return s; }
    function tree(v) { v = Array.prototype.slice.call(v); while (v.length > 1) { var nx = []; for (var i = 0; i + 1 < v.length; i += 2) nx.push(f(v[i] + v[i + 1])); if (v.length % 2) nx.push(v[v.length - 1]); v = nx; } return v.length ? v[0] : 0; }
    function lanes(x, V) { var acc = new Array(V).fill(0); for (var i = 0; i < x.length; i++) acc[i % V] = f(acc[i % V] + x[i]); return tree(acc); }
    function kahan(x) { var s = 0, cc = 0; for (var i = 0; i < x.length; i++) { var y = f(x[i] - cc), t = f(s + y); cc = f(f(t - s) - y); s = t; } return s; }
    function exact(x) { // Neumaier in float64 on float32 inputs: exact to well below float32 resolution here
      var s = 0, cc = 0; for (var i = 0; i < x.length; i++) { var t = s + x[i]; cc += Math.abs(s) >= Math.abs(x[i]) ? (s - t) + x[i] : (x[i] - t) + s; s = t; } return s + cc;
    }
    function ulp(v) { var a = Math.abs(f(v)); if (a === 0) return 1.4e-45; var e = Math.floor(Math.log2(a)); return Math.pow(2, e - 23); }
    function draw() {
      var x = data(), ex = exact(x), u = ulp(ex);
      var res = [['sequential (the C source)', seq(x)], ['V = 4 lanes + tree', lanes(x, 4)], ['V = 8 lanes + tree', lanes(x, 8)],
        ['V = 16 lanes + tree', lanes(x, 16)], ['pairwise tree', tree(x)], ['Kahan compensated', kahan(x)]];
      clear(ctx, c);
      var errs = res.map(function (r) { return Math.abs(r[1] - ex) / u; }), mx = Math.max.apply(null, errs.concat([1]));
      res.forEach(function (r, k) {
        var y = 16 + k * 34, w = 300 * Math.log10(1 + errs[k]) / Math.log10(1 + mx);
        txt(ctx, r[0], 10, y + 8, C.text, 10); rect(ctx, 200, y, Math.max(2, w), 16, LANE[k]);
        txt(ctx, fmt(errs[k], 1) + ' ulp', 206 + Math.max(2, w), y + 8, C.mid, 10);
      });
      out.textContent = 'exact sum          ' + ex.toPrecision(12) + '\n' + res.map(function (r) { return (r[0] + '                          ').slice(0, 26) + r[1].toPrecision(10); }).join('\n');
    }
    [nS, dS].forEach(function (x) { x.addEventListener('change', draw); });
    reB.onclick = function () { seed = (seed * 17 + 5) % 100000; draw(); };
    draw();
  }

  // ------------------------------------------------------------ 7. systolic: output-stationary array (vzlab.systolic)
  function labSystolic(root) {
    var body = frame(root, 'Lab · A systolic array, one clock at a time');
    var c = canvas(560, 330); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var sS = select(ct, 'array R×C', [3, 4], 4), kS = select(ct, 'K (reduction length)', [3, 4, 5, 6], 4);
    var ct2 = controls(p), stepB = button(ct2, 'Step'), playB = button(ct2, 'Play'), resetB = button(ct2, 'Reset');
    var out = readout(p);
    hint(p, 'A[i,k] enters row i from the left, B[k,j] enters column j from the top, each skewed one clock per row/column. They meet in PE(i,j) at clock t = i + j + k.');
    var A, B, Cm, t, timer = null, busy;
    function init() {
      var R = +sS.value, K = +kS.value, r = rng(11 + R * 7 + K), i, j;
      A = []; B = []; Cm = [];
      for (i = 0; i < R; i++) { A.push([]); for (var k = 0; k < K; k++) A[i].push(Math.floor(r() * 7) - 3); }
      for (k = 0; k < K; k++) { B.push([]); for (j = 0; j < R; j++) B[k].push(Math.floor(r() * 7) - 3); }
      for (i = 0; i < R; i++) { Cm.push([]); for (j = 0; j < R; j++) Cm[i].push(0); }
      t = 0; busy = 0; draw();
    }
    function total() { var R = +sS.value; return +kS.value + 2 * R - 2; }
    function step() {
      var R = +sS.value, K = +kS.value;
      if (t >= total()) { if (timer) { clearInterval(timer); timer = null; } return; }
      for (var i = 0; i < R; i++) for (var j = 0; j < R; j++) { var k = t - i - j; if (k >= 0 && k < K) { Cm[i][j] += A[i][k] * B[k][j]; busy++; } }
      t++; draw();
    }
    function draw() {
      clear(ctx, c);
      var R = +sS.value, K = +kS.value, cell = 52, x0 = 150, y0 = 110, i, j, k;
      for (i = 0; i < R; i++) for (j = 0; j < R; j++) {
        var kk = t - 1 - i - j, active = t > 0 && kk >= 0 && kk < K, x = x0 + j * cell, y = y0 + i * cell;
        rect(ctx, x, y, cell - 8, cell - 8, active ? 'rgba(34,211,238,0.25)' : '#151515', active ? C.acc : '#3a3a3a');
        txt(ctx, String(Cm[i][j]), x + (cell - 8) / 2, y + (cell - 8) / 2, active ? '#fff' : C.mid, 13, 'center');
        if (active) txt(ctx, 'k=' + kk, x + (cell - 8) / 2, y + cell - 16, C.acc, 8, 'center');
      }
      // left queues of A (skewed)
      for (i = 0; i < R; i++) for (k = 0; k < K; k++) {
        var slot = k + i - t; // distance from the array edge in clocks
        if (slot < 0) continue; var xx = x0 - 22 - slot * 22;
        if (xx < 4) continue; rect(ctx, xx, y0 + i * cell + 12, 20, 18, '#2a1f3d', '#5b4a7a'); txt(ctx, String(A[i][k]), xx + 10, y0 + i * cell + 21, C.purple, 10, 'center');
      }
      for (j = 0; j < R; j++) for (k = 0; k < K; k++) {
        var sl = k + j - t; if (sl < 0) continue; var yy = y0 - 22 - sl * 20;
        if (yy < 2) continue; rect(ctx, x0 + j * cell + 12, yy, 20, 18, '#3d2a1f', '#7a5b4a'); txt(ctx, String(B[k][j]), x0 + j * cell + 22, yy + 9, C.yellow, 10, 'center');
      }
      txt(ctx, 'A →', 8, y0 + 14, C.purple, 11); txt(ctx, 'B ↓', x0 + R * cell + 8, 14, C.yellow, 11);
      var ok = '';
      if (t >= total()) {
        var good = true;
        for (i = 0; i < R; i++) for (j = 0; j < R; j++) { var s = 0; for (k = 0; k < K; k++) s += A[i][k] * B[k][j]; if (s !== Cm[i][j]) good = false; }
        ok = '\nresult equals A @ B   ' + (good ? 'yes' : 'NO');
      }
      out.textContent = 'clock                 ' + t + ' / ' + total() + '\nMACs per clock (peak) ' + (R * R) +
        '\nMACs done             ' + busy + ' / ' + (R * R * K) + '\nutilisation so far    ' + (t ? fmt(100 * busy / (R * R * t), 1) : '—') + ' %' +
        '\nfill + drain clocks   ' + (2 * R - 2) + ' (why big arrays want long K)' + ok;
    }
    stepB.onclick = step;
    playB.onclick = function () { if (timer) { clearInterval(timer); timer = null; return; } if (t >= total()) init(); timer = setInterval(step, 600); };
    resetB.onclick = function () { if (timer) { clearInterval(timer); timer = null; } init(); };
    [sS, kS].forEach(function (x) { x.addEventListener('change', init); });
    init();
  }

  // ------------------------------------------------------------ 8. broadcast: shapes (vzlab.broadcast)
  function labBroadcast(root) {
    var body = frame(root, 'Lab · Broadcasting — right-align, then 1 stretches');
    var p = el('div', { style: 'flex:1 1 100%' }); body.appendChild(p);
    var ct = controls(p);
    var aI = el('input', { type: 'text', value: '4, 1', size: 12 }), bI = el('input', { type: 'text', value: '3', size: 12 });
    [aI, bI].forEach(function (x) { x.style.cssText = 'font:inherit;font-size:.8rem;background:var(--bg-elevated);color:var(--text-primary);border:1px solid var(--border);border-radius:4px;padding:2px 6px'; });
    var la = el('label', null, 'shape A '); la.appendChild(aI); ct.appendChild(la);
    var lb = el('label', null, 'shape B '); lb.appendChild(bI); ct.appendChild(lb);
    var pre = select(ct, 'examples', [['', '—'], ['4, 1|3', 'column + row → outer'], ['8, 1, 6, 1|7, 1, 5', 'NumPy docs example'], ['3, 4|3', 'the classic error'],
      ['32, 1, 128|1, 64, 128', 'pairwise (n,1,d) − (1,m,d)'], ['2, 3|', 'scalar']], '');
    var grid = el('div'); p.appendChild(grid);
    var out = readout(p);
    hint(p, 'A stretched axis is a stride-0 view: no copy is made until an operation writes the result.');
    function parse(s) { s = s.trim(); if (!s) return []; return s.split(/[\s,x×]+/).filter(Boolean).map(Number); }
    function draw() {
      var A = parse(aI.value), Bs = parse(bI.value), nd = Math.max(A.length, Bs.length);
      var pa = [], pb = [], i; for (i = 0; i < nd - A.length; i++) pa.push(1); pa = pa.concat(A);
      for (i = 0; i < nd - Bs.length; i++) pb.push(1); pb = pb.concat(Bs);
      var res = [], err = null, rows = [];
      for (i = 0; i < nd; i++) {
        var a = pa[i], b = pb[i], r = a === b ? a : a === 1 ? b : b === 1 ? a : null;
        if (r === null && !err) err = 'axis ' + (i - nd) + ': ' + a + ' vs ' + b + ' — only equal sizes or 1 may meet';
        res.push(r);
        rows.push('<tr><td>' + (i - nd) + '</td><td>' + (i < nd - A.length ? '<i style="color:#777">1 (added)</i>' : a) + (a === 1 && r > 1 ? ' → stretched' : '') +
          '</td><td>' + (i < nd - Bs.length ? '<i style="color:#777">1 (added)</i>' : b) + (b === 1 && r > 1 ? ' → stretched' : '') + '</td><td>' + (r === null ? '<span class="chip no">✗</span>' : r) + '</td></tr>');
      }
      var html = '<table><thead><tr><th>axis</th><th>A</th><th>B</th><th>result</th></tr></thead><tbody>' + rows.join('') + '</tbody></table>';
      if (!err && nd === 2 && res[0] <= 6 && res[1] <= 8) {
        html += '<div class="sheets" style="margin:.5rem 0">' + sheet('A', pa, res) + '<span class="op">+</span>' + sheet('B', pb, res) + '<span class="op">=</span>' + sheet('A+B', res, res, true) + '</div>';
      }
      grid.innerHTML = html;
      out.textContent = err ? 'ValueError: ' + err : 'result shape (' + res.join(', ') + (res.length === 1 ? ',' : '') + ')';
    }
    function sheet(name, shp, res, full) {
      var h = '<table class="sheet"><caption>' + name + ' (' + shp.join(',') + ')</caption><tr><th></th>';
      for (var j = 0; j < res[1]; j++) h += '<th>' + String.fromCharCode(65 + j) + '</th>';
      h += '</tr>';
      for (var i = 0; i < res[0]; i++) {
        h += '<tr><th>' + (i + 1) + '</th>';
        for (j = 0; j < res[1]; j++) {
          var real = full || ((shp[0] === 1 ? i === 0 : true) && (shp[1] === 1 ? j === 0 : true));
          var si = shp[0] === 1 ? 0 : i, sj = shp[1] === 1 ? 0 : j;
          h += '<td class="' + (real ? (full ? '' : 's') : 'g') + '">' + (full ? '·' : name.toLowerCase() + si + sj) + '</td>';
        }
        h += '</tr>';
      }
      return h + '</table>';
    }
    [aI, bI].forEach(function (x) { x.addEventListener('input', draw); });
    pre.addEventListener('change', function () { if (!pre.value) return; var s = pre.value.split('|'); aI.value = s[0]; bI.value = s[1]; draw(); });
    draw();
  }

  // ------------------------------------------------------------ 9. amdahl (vzlab.perfmodel.amdahl)
  function labAmdahl(root) {
    var body = frame(root, 'Lab · Amdahl — the part you did not vectorize sets the ceiling');
    var c = canvas(560, 260); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var fS = slider(ct, 'vectorizable fraction f', 0, 1, 0.01, 0.9), eS = slider(ct, 'lane efficiency', 0.1, 1, 0.05, 0.7);
    var vS = select(controls(p), 'lanes V', [4, 8, 16, 32, 64], 16);
    var out = readout(p);
    hint(p, 'Lane efficiency folds in tails, masks, shuffles and memory stalls: effective vector speedup = V × efficiency.');
    function draw() {
      clear(ctx, c);
      var f = +fS.value, e = +eS.value, V = +vS.value, L = 50, R = 330, T = 16, B = 230;
      var maxS = Math.min(64, 1 / Math.max(1e-6, 1 - f)) * 1.1 + 1;
      function X(v) { return L + Math.log2(v) / 6 * (R - L); }
      function Y(s) { return B - (s - 1) / (maxS - 1) * (B - T); }
      ctx.strokeStyle = C.grid; [1, 2, 4, 8, 16, 32, 64].forEach(function (v) { ctx.beginPath(); ctx.moveTo(X(v), T); ctx.lineTo(X(v), B); ctx.stroke(); txt(ctx, String(v), X(v), B + 12, C.mid, 9, 'center'); });
      txt(ctx, 'lanes V', (L + R) / 2, B + 26, C.mid, 10, 'center'); txt(ctx, 'speedup', 6, T, C.mid, 9);
      ctx.strokeStyle = C.acc; ctx.lineWidth = 2; ctx.beginPath();
      for (var lv = 0; lv <= 6.001; lv += 0.05) { var v = Math.pow(2, lv), s = 1 / ((1 - f) + f / Math.max(1, v * e)); if (lv === 0) ctx.moveTo(X(v), Y(s)); else ctx.lineTo(X(v), Y(s)); }
      ctx.stroke();
      if (f < 1) { var cap = 1 / (1 - f); if (cap < maxS) { ctx.strokeStyle = C.red; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(L, Y(cap)); ctx.lineTo(R, Y(cap)); ctx.stroke(); ctx.setLineDash([]); txt(ctx, 'limit 1/(1−f) = ' + fmt(cap, 1) + '×', R - 2, Y(cap) - 8, C.red, 9, 'right'); } }
      var sv = 1 / ((1 - f) + f / Math.max(1, V * e));
      ctx.fillStyle = C.yellow; ctx.beginPath(); ctx.arc(X(V), Y(sv), 4, 0, 7); ctx.fill();
      // time bars
      var bx = 360, bw = 180; txt(ctx, 'time, before', bx, 30, C.mid, 10); rect(ctx, bx, 40, bw * (1 - f), 22, '#777'); rect(ctx, bx + bw * (1 - f), 40, bw * f, 22, C.acc);
      txt(ctx, 'time, after', bx, 90, C.mid, 10); var t2 = (1 - f) + f / Math.max(1, V * e);
      rect(ctx, bx, 100, bw * (1 - f), 22, '#777'); rect(ctx, bx + bw * (1 - f), 100, bw * f / Math.max(1, V * e), 22, C.acc);
      txt(ctx, 'grey: scalar part  cyan: vector part', bx, 140, C.mid, 9);
      out.textContent = 'effective vector speedup  ' + fmt(V * e, 1) + '×\nwhole-program speedup     ' + fmt(sv, 2) + '×' +
        '\nscalar part of new time   ' + fmt(100 * (1 - f) / t2, 0) + ' %';
    }
    [fS, eS, vS].forEach(function (x) { x.addEventListener('input', draw); x.addEventListener('change', draw); });
    draw();
  }

  // ------------------------------------------------------------ 10. atlas: vector ISAs and matrix engines
  function labAtlas(root) {
    var body = frame(root, 'Vector ISA Atlas — every instruction set and engine in the book');
    var wrapAll = el('div', { style: 'flex:1 1 100%;min-width:0' }); body.appendChild(wrapAll);
    var f = el('div', { 'class': 'atlas-filters' }); wrapAll.appendChild(f);
    var kind = select(f, 'kind', [['', 'all'], ['CPU SIMD', 'CPU SIMD'], ['CPU vector-length agnostic', 'CPU VLA'], ['GPU SIMT', 'GPU SIMT'], ['matrix engine', 'matrix engine'],
      ['systolic array', 'systolic array'], ['DSP vector', 'DSP vector'], ['vector machine', 'vector machine']], '');
    var q = el('input', { type: 'text', placeholder: 'search name, vendor, types…', size: 22 });
    q.style.cssText = 'font:inherit;font-size:.78rem;background:var(--bg-elevated);color:var(--text-primary);border:1px solid var(--border);border-radius:4px;padding:2px 6px';
    f.appendChild(q);
    var c = canvas(640, 280); c.style.maxWidth = '100%'; wrapAll.appendChild(c); var ctx = c.getContext('2d');
    var tip = el('div', { 'class': 'lab-readout', style: 'min-height:3.2em;margin:.5rem 0' }, 'Hover a point.'); wrapAll.appendChild(tip);
    var wrap = el('div', { 'class': 'atlas-table-wrap' }); wrapAll.appendChild(wrap);
    var data = [], pts = [], sortKey = 'year', sortDir = 1;
    var KC = { 'CPU SIMD': C.acc, 'CPU vector-length agnostic': C.green, 'GPU SIMT': C.purple, 'matrix engine': C.yellow, 'systolic array': '#fb923c', 'DSP vector': C.pink, 'vector machine': C.blue };
    function rows() {
      var s = q.value.toLowerCase();
      return data.filter(function (d) { return (!kind.value || d.kind === kind.value) && (!s || JSON.stringify(d).toLowerCase().indexOf(s) >= 0); });
    }
    function draw() {
      var rs = rows(); clear(ctx, c); pts = [];
      var L = 56, R = 560, T = 14, B = 246, y0 = 1970, y1 = 2027;
      function X(y) { return L + (y - y0) / (y1 - y0) * (R - L); }
      function Y(b) { return B - (Math.log2(b) - 5) / (21 - 5) * (B - T); }
      ctx.strokeStyle = C.grid;
      [64, 128, 256, 512, 1024, 4096, 65536, 1048576].forEach(function (b) { ctx.beginPath(); ctx.moveTo(L, Y(b)); ctx.lineTo(R, Y(b)); ctx.stroke(); txt(ctx, b >= 65536 ? (b / 1024) + 'K' : String(b), L - 6, Y(b), C.mid, 9, 'right'); });
      [1975, 1985, 1995, 2005, 2015, 2025].forEach(function (y) { txt(ctx, String(y), X(y), B + 12, C.mid, 9, 'center'); });
      txt(ctx, 'bits processed by one instruction (log scale)', L, T - 4, C.mid, 9);
      rs.forEach(function (d) {
        var x = X(d.year), y = Y(d.width_bits); pts.push({ x: x, y: y, d: d });
        ctx.fillStyle = KC[d.kind] || '#888'; ctx.beginPath();
        if (d.vla) { ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 6); ctx.lineTo(x - 6, y); ctx.closePath(); } else ctx.arc(x, y, 5, 0, 7);
        ctx.fill();
      });
      var lx = R + 6, ly = 20; Object.keys(KC).forEach(function (k) { ctx.fillStyle = KC[k]; ctx.beginPath(); ctx.arc(lx + 4, ly, 4, 0, 7); ctx.fill(); txt(ctx, k.replace('vector-length agnostic', 'VLA'), lx + 12, ly, C.mid, 9); ly += 16; });
      txt(ctx, '◆ = vector-length agnostic', lx, ly + 6, C.mid, 9);
      rs.sort(function (a, b) { var x = a[sortKey], y = b[sortKey]; return (x > y ? 1 : x < y ? -1 : 0) * sortDir; });
      var cols = [['name', 'Name'], ['vendor', 'Vendor'], ['kind', 'Kind'], ['year', 'Year'], ['width_bits', 'Bits'], ['vla', 'VLA'], ['predication', 'Masking'], ['gather', 'Gather'], ['scatter', 'Scatter'], ['ml_types', 'ML types']];
      var h = '<table><thead><tr>' + cols.map(function (k) { return '<th data-k="' + k[0] + '">' + k[1] + (sortKey === k[0] ? (sortDir > 0 ? ' ▲' : ' ▼') : '') + '</th>'; }).join('') + '</tr></thead><tbody>';
      rs.forEach(function (d) { h += '<tr>' + cols.map(function (k) { var v = d[k[0]]; if (v === true) v = 'yes'; if (v === false) v = 'no'; return '<td>' + v + '</td>'; }).join('') + '</tr>'; });
      wrap.innerHTML = h + '</tbody></table>';
      Array.prototype.forEach.call(wrap.querySelectorAll('th'), function (th) { th.addEventListener('click', function () { var k = th.getAttribute('data-k'); if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = 1; } draw(); }); });
    }
    c.addEventListener('mousemove', function (ev) {
      var b = c.getBoundingClientRect(), x = (ev.clientX - b.left) * c._w / b.width, y = (ev.clientY - b.top) * c._h / b.height, best = null, bd = 14;
      pts.forEach(function (p) { var d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; best = p; } });
      if (best) { var d = best.d; tip.textContent = d.name + ' — ' + d.vendor + ', ' + d.year + ' (' + d.first_silicon + ')\n' + d.width_bits + ' bits · ' + d.kind + (d.matrix ? ' · ' + d.matrix : '') + '\n' + d.note + '\nsource: ' + d.source; }
    });
    [kind].forEach(function (x) { x.addEventListener('change', draw); }); q.addEventListener('input', draw);
    var base = (window.VZ_BASEURL || '').replace(/\/$/, '');
    fetch(base + '/assets/data/isas.json').then(function (r) { return r.json(); }).then(function (j) { data = j; draw(); })
      .catch(function () { tip.textContent = 'Could not load assets/data/isas.json — open the published site, or see Appendix D for the same rows as a table.'; });
  }

  var LABS = { lanes: labLanes, deps: labDeps, stride: labStride, mask: labMask, roofline: labRoofline, reduce: labReduce,
               systolic: labSystolic, broadcast: labBroadcast, amdahl: labAmdahl, atlas: labAtlas };
  function init() {
    Array.prototype.forEach.call(document.querySelectorAll('.lab[data-lab]'), function (root) {
      var f = LABS[root.getAttribute('data-lab')];
      if (f) { try { f(root); } catch (e) { root.textContent = 'Lab failed to load: ' + e; } }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
