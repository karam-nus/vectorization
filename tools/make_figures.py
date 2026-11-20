#!/usr/bin/env python3
"""Generate every static figure in assets/images/ from tools/measurements/results.json.

    python tools/make_figures.py

Pure-Python SVG (no plotting library), dark surface to match the site. Colours come from a
palette validated for colour-vision deficiency on the #111111 surface; every mark carries a
<title> so hovering the SVG (opened directly) shows its value. Never hand-edit the outputs.
"""
import json
import math
import os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = json.load(open(os.path.join(HERE, "measurements", "results.json")))
E = RES["experiments"]
OUT = os.path.join(ROOT, "assets", "images")
os.makedirs(OUT, exist_ok=True)

SURF, GRID, AXIS = "#111111", "#262626", "#3a3a3a"
T1, T2, T3 = "#e8e8e8", "#a3a3a3", "#6b6b6b"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
MACHINE = "Xeon (Emerald Rapids), 1 core, gcc 13.3"


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, fill=T2, anchor="start", weight=400, mono=False, extra=""):
        fam = MONO if mono else FONT
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{escape(str(s))}</text>')

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>')

    def hbar(self, x0, y, length, thick, fill, title=""):
        """Horizontal bar: square at the baseline (x0), 4px rounded data-end."""
        r = min(4, length / 2, thick / 2)
        x1 = x0 + length
        d = (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
             f"V{y + thick - r:.1f} Q{x1:.1f},{y + thick:.1f} {x1 - r:.1f},{y + thick:.1f} H{x0:.1f} Z")
        self.add(f'<path d="{d}" fill="{fill}"><title>{escape(title)}</title></path>')

    def vbar(self, x, y_base, height, thick, fill, title=""):
        """Vertical bar growing up from y_base: square at the baseline, rounded top."""
        r = min(4, height / 2, thick / 2)
        y1 = y_base - height
        d = (f"M{x:.1f},{y_base:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} "
             f"H{x + thick - r:.1f} Q{x + thick:.1f},{y1:.1f} {x + thick:.1f},{y1 + r:.1f} V{y_base:.1f} Z")
        self.add(f'<path d="{d}" fill="{fill}"><title>{escape(title)}</title></path>')

    def dot(self, x, y, fill, r=4.5, title="", hollow=False):
        if hollow:
            self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{SURF}" stroke="{fill}" stroke-width="2">'
                     f'<title>{escape(title)}</title></circle>')
        else:
            self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{SURF}" stroke-width="2">'
                     f'<title>{escape(title)}</title></circle>')

    def poly(self, pts, stroke, width=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.add(f'<polyline points="{p}" fill="none" stroke="{stroke}" stroke-width="{width}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def legend(self, x, y, items, gap=18):
        cx = x
        for label, color in items:
            self.add(f'<rect x="{cx:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
            self.text(cx + 15, y + 4, label, 11, T2)
            cx += 15 + 6.3 * len(label) + gap

    def save(self, name, title, desc):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="t d">\n'
               f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>\n'
               f'<rect width="{self.w}" height="{self.h}" rx="10" fill="{SURF}"/>\n{body}\n</svg>\n')
        with open(os.path.join(OUT, name), "w") as f:
            f.write(svg)
        print("wrote", name)


def header(s, title, sub):
    s.text(24, 32, title, 16, T1, weight=600)
    s.text(24, 52, sub, 12, T3)


def find(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    raise KeyError(kw)


# ---------------------------------------------------------------- 1. saxpy at four memory levels
def fig_saxpy_levels():
    rows = E["saxpy_memory_levels"]
    variants = [("scalar", "scalar"), ("sse2", "SSE2 128-bit"), ("avx2", "AVX2 256-bit"), ("avx512", "AVX-512")]
    levels = [(4096, "L1", "16 KB / array"), (65536, "L2", "256 KB"), (1 << 20, "L3", "4 MB"), (1 << 26, "DRAM", "256 MB")]
    s = Svg(860, 420)
    header(s, "Wider registers pay off only while the data is close to the core",
           f"saxpy y = a·x + y, fp32, GB/s moved · {MACHINE}")
    L, R, T, B = 70, 830, 100, 360
    vmax = 350
    for g in range(0, vmax + 1, 50):
        y = B - g / vmax * (B - T)
        s.line(L, y, R, y, GRID)
        s.text(L - 8, y + 4, g, 11, T3, "end")
    s.text(L - 8, T - 14, "GB/s", 11, T3, "end")
    gw = (R - L) / len(levels)
    bw = 24
    for gi, (n, lvl, size) in enumerate(levels):
        gx = L + gi * gw + (gw - (bw * 4 + 2 * 3)) / 2
        for vi, (v, vname) in enumerate(variants):
            r = find(rows, variant=v, n=n)
            h = r["gbps"] / vmax * (B - T)
            s.vbar(gx + vi * (bw + 2), B, h, bw, SERIES[vi], f"{vname}, {lvl}: {r['gbps']:.1f} GB/s ({r['ns_per_elem']:.3f} ns/element)")
            if v in ("scalar", "avx512"):
                s.text(gx + vi * (bw + 2) + bw / 2, B - h - 6, f"{r['gbps']:.0f}", 11, T1, "middle", 600)
        s.text(L + gi * gw + gw / 2, B + 20, lvl, 13, T1, "middle", 600)
        sc, av = find(rows, variant="scalar", n=n), find(rows, variant="avx512", n=n)
        s.text(L + gi * gw + gw / 2, B + 37, f"{size} · AVX-512 {av['gbps'] / sc['gbps']:.1f}× scalar", 11, T3, "middle")
    s.line(L, B, R, B, AXIS)
    s.legend(L, 80, [(v[1], SERIES[i]) for i, v in enumerate(variants)])
    s.save("saxpy_levels.svg", "saxpy bandwidth by vector width and memory level",
           "Grouped bars: in L1 AVX-512 moves about nine times the bytes of scalar code; in DRAM all vector widths "
           "converge near 20 GB/s.")


# ---------------------------------------------------------------- 2. speedup heatmap, kernels x variants (L1)
def fig_width_heatmap():
    rows = E["width_ladder_L1"]
    kernels = [("saxpy", "saxpy  y=a·x+y", "map"), ("step_soa", "particle step, SoA", "map, 3 streams"),
               ("leaky", "leaky ReLU", "branch → select"), ("step_aos", "particle step, AoS", "interleaved fields"),
               ("dot", "dot product", "fp reduction"), ("prefix", "prefix sum", "loop-carried")]
    variants = [("scalar", "scalar"), ("sse2", "SSE2"), ("avx2", "AVX2"), ("avx512", "AVX-512"), ("avx512_fast", "AVX-512\n-ffast-math")]
    s = Svg(860, 430)
    header(s, "Same compiler, same source: speedup over scalar depends on the loop, not the register",
           f"fp32, n = 4096 (in L1), gcc -O3 per target · {MACHINE}")
    L, T = 250, 108
    cw, ch = 112, 44
    for vi, (_, vname) in enumerate(variants):
        for li, part in enumerate(vname.split("\n")):
            s.text(L + vi * cw + cw / 2, T - 18 + li * 13 - (6 if "\n" in vname else 0), part, 12, T1 if li == 0 else T3, "middle", 600 if li == 0 else 400)
    for ki, (k, kname, kind) in enumerate(kernels):
        y = T + ki * ch
        s.text(24, y + ch / 2 - 2, kname, 13, T1, weight=600)
        s.text(24, y + ch / 2 + 14, kind, 11, T3)
        base = find(rows, variant="scalar", kernel=k)["sec"]
        for vi, (v, vname) in enumerate(variants):
            r = find(rows, variant=v, kernel=k)
            sp = base / r["sec"]
            # diverging around 1x on a log scale: blue = faster, red = slower, grey = no change
            t = max(-1.0, min(1.0, math.log2(sp) / math.log2(16)))
            if abs(math.log2(sp)) < 0.2:
                fill = "#2b2b2b"
            elif t > 0:
                a = 0.25 + 0.75 * t
                fill = f"rgba(57,135,229,{a:.2f})"
            else:
                a = 0.35 + 0.65 * (-t / 0.5 if -t < 0.5 else 1)
                fill = f"rgba(230,103,103,{min(1, a):.2f})"
            x = L + vi * cw
            s.add(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{fill}">'
                  f'<title>{escape(kname)}, {escape(vname.replace(chr(10), " "))}: {sp:.2f}× scalar ({r["ns_per_elem"]:.3f} ns/element)</title></rect>')
            s.text(x + cw / 2, y + ch / 2 + 5, f"{sp:.1f}×", 13, "#ffffff" if t > 0.45 else T1, "middle", 600, mono=True)
    yb = T + len(kernels) * ch + 26
    s.add(f'<rect x="{L}" y="{yb - 9}" width="12" height="12" rx="2" fill="rgba(57,135,229,0.9)"/>')
    s.text(L + 18, yb + 1, "faster than scalar", 11, T2)
    s.add(f'<rect x="{L + 150}" y="{yb - 9}" width="12" height="12" rx="2" fill="#2b2b2b"/>')
    s.text(L + 168, yb + 1, "no change (not vectorized, or bound elsewhere)", 11, T2)
    s.add(f'<rect x="{L + 460}" y="{yb - 9}" width="12" height="12" rx="2" fill="rgba(230,103,103,0.9)"/>')
    s.text(L + 478, yb + 1, "slower than scalar", 11, T2)
    s.save("width_heatmap.svg", "Speedup over scalar for six loops across five compile targets",
           "Heatmap. saxpy reaches 9x with AVX-512; leaky ReLU vectorizes only with AVX-512 masking; the strict-IEEE "
           "dot product gets slower with AVX-512 and fast only with -ffast-math; the prefix sum never vectorizes.")


# ---------------------------------------------------------------- 3. stride sweep
def fig_stride():
    rows = E["stride_sweep"]
    strides = [1, 2, 4, 8, 16, 32]
    series = [("scalar", "strided", "scalar loop"), ("avx512", "strided", "AVX-512 auto-vectorized"),
              ("avx512", "strided_gather", "AVX-512 vgatherdps intrinsics")]
    s = Svg(800, 420)
    header(s, "Stride decides the cost; the vector width does not",
           f"y[i] = 2·x[i·s], fp32, 1 M outputs, ns per useful element (log scale) · {MACHINE}")
    L, R, T, B = 80, 700, 96, 350
    ymin, ymax = math.log10(0.2), math.log10(20)

    def X(st):
        return L + math.log2(st) / 5 * (R - L)

    def Y(v):
        return B - (math.log10(v) - ymin) / (ymax - ymin) * (B - T)
    for v in (0.2, 0.5, 1, 2, 5, 10, 20):
        s.line(L, Y(v), R, Y(v), GRID)
        s.text(L - 8, Y(v) + 4, v, 11, T3, "end")
    for st in strides:
        s.line(X(st), B, X(st), B + 4, AXIS)
        s.text(X(st), B + 18, st, 11, T2, "middle")
    s.text((L + R) / 2, B + 36, "stride s (elements)", 11, T3, "middle")
    s.line(L, B, R, B, AXIS)
    # reference: cache lines per element
    s.line(X(16), T, X(16), B, "#4a4a4a", 1)
    s.text(X(16) - 6, T + 12, "s = 16: one 64-byte line", 11, T2, "end")
    s.text(X(16) - 6, T + 27, "per element (4 of 64 B used)", 11, T2, "end")
    s.text(R + 10, Y(8.3) + 4, "≈ 8 ns", 11, T2)
    for si, (v, k, name) in enumerate(series):
        pts = []
        for st in strides:
            r = find(rows, variant=v, kernel=k, stride=st)
            pts.append((X(st), Y(r["ns_per_elem"]), r))
        s.poly([(x, y) for x, y, _ in pts], SERIES[si])
        for x, y, r in pts:
            s.dot(x, y, SERIES[si], 4, f"{name}, stride {r['stride']}: {r['ns_per_elem']:.3f} ns/element")
    s.legend(L, 78, [(n, SERIES[i]) for i, (_, _, n) in enumerate(series)])
    s.save("stride_sweep.svg", "Cost per element versus access stride",
           "Three nearly identical lines rising from about 0.3 ns at stride 1 to about 8 ns at stride 32: the number of "
           "cache lines touched, not the instruction set, sets the cost.")


# ---------------------------------------------------------------- 4. dot product accumulators
def fig_dot():
    rows = E["dot_accumulators_L1"]
    items = [("scalar", "dot", "scalar, in order"), ("avx512", "dot", "gcc AVX-512, strict IEEE"),
             ("avx512_fast", "dot", "gcc AVX-512, -ffast-math"), ("avx512", "dot_avx512_1acc", "intrinsics, 1 accumulator"),
             ("avx512", "dot_avx512_2acc", "intrinsics, 2 accumulators"), ("avx512", "dot_avx512_4acc", "intrinsics, 4 accumulators"),
             ("avx512", "dot_avx512_8acc", "intrinsics, 8 accumulators")]
    s = Svg(860, 400)
    header(s, "A reduction is a latency chain until you give it independent accumulators",
           f"dot product, fp32, n = 4096 (in L1), GFLOP/s · {MACHINE}")
    L, R, T = 250, 760, 84
    vmax = 80
    for g in range(0, vmax + 1, 20):
        x = L + g / vmax * (R - L)
        s.line(x, T - 6, x, T + len(items) * 38, GRID)
        s.text(x, T + len(items) * 38 + 16, g, 11, T3, "middle")
    s.text(R, T + len(items) * 38 + 32, "GFLOP/s", 11, T3, "end")
    base = find(rows, variant="scalar", kernel="dot")["gflops"]
    for i, (v, k, name) in enumerate(items):
        r = find(rows, variant=v, kernel=k)
        y = T + i * 38
        s.text(L - 12, y + 16, name, 12, T1, "end")
        fill = SERIES[0] if r["gflops"] >= base * 0.95 else SERIES[7]
        s.hbar(L, y + 4, r["gflops"] / vmax * (R - L), 22, fill, f"{name}: {r['gflops']:.1f} GFLOP/s, {r['gflops'] / base:.2f}× scalar")
        s.text(L + r["gflops"] / vmax * (R - L) + 8, y + 19, f"{r['gflops']:.1f}  ({r['gflops'] / base:.2f}×)", 11, T2, mono=True)
    s.save("dot_accumulators.svg", "Dot product throughput by implementation",
           "Bars: strict-IEEE AVX-512 code is slower than scalar; -ffast-math and one accumulator reach about 27 GFLOP/s; "
           "four accumulators reach about 70 GFLOP/s.")


# ---------------------------------------------------------------- 5. the pairwise-distance ladder
def fig_ladder():
    py = {r["rung"]: r for r in E["pdist_python"]}
    c = E["pdist_c"]
    rungs = [
        ("Python lists, 3 nested loops", py["python_loops"]["ns_per_pair_dim"], "Python"),
        ("NumPy broadcast (n, m, d) temp", py["numpy_broadcast"]["ns_per_pair_dim"], "NumPy"),
        ("NumPy, loop over rows", py["numpy_rowloop"]["ns_per_pair_dim"], "NumPy"),
        ("C naive, -O2 scalar", find(c, variant="scalar", kernel="pdist_naive")["ns_per_pair_dim"], "C"),
        ("C naive, -O3 AVX-512 (strict FP)", find(c, variant="avx512", kernel="pdist_naive")["ns_per_pair_dim"], "C"),
        ("C naive, -O3 AVX-512 -ffast-math", find(c, variant="avx512_fast", kernel="pdist_naive")["ns_per_pair_dim"], "C"),
        ("NumPy ‖x‖²+‖y‖²−2XYᵀ (one GEMM)", py["numpy_gemm"]["ns_per_pair_dim"], "NumPy"),
        ("C loops reordered, AVX-512", find(c, variant="avx512", kernel="pdist_outer")["ns_per_pair_dim"], "C"),
    ]
    s = Svg(860, 470)
    header(s, "One problem, eight rungs: pairwise squared distances, 1024 × 1024 points, d = 64",
           f"ns per (pair × dimension), log scale, shorter is faster · {MACHINE}, NumPy 2.5 (single core)")
    L, R, T = 300, 740, 92
    lo, hi = math.log10(0.03), math.log10(100)
    for v in (0.03, 0.1, 0.3, 1, 3, 10, 30, 100):
        x = L + (math.log10(v) - lo) / (hi - lo) * (R - L)
        s.line(x, T - 6, x, T + len(rungs) * 40, GRID)
        s.text(x, T + len(rungs) * 40 + 16, v, 11, T3, "middle")
    s.text(R, T + len(rungs) * 40 + 32, "ns per pair-dimension", 11, T3, "end")
    colors = {"Python": SERIES[1], "NumPy": SERIES[2], "C": SERIES[0]}
    first = rungs[0][1]
    for i, (name, v, lang) in enumerate(rungs):
        y = T + i * 40
        s.text(L - 12, y + 17, f"{i + 1}. {name}", 12, T1, "end")
        # bars on a log axis grow from the left edge (0.03) — length encodes log time
        length = (math.log10(v) - lo) / (hi - lo) * (R - L)
        s.hbar(L, y + 5, length, 22, colors[lang], f"{name}: {v:.4f} ns, {first / v:.0f}× faster than rung 1")
        s.text(L + length + 8, y + 20, f"{v:.3g} ns · {first / v:,.0f}×", 11, T2, mono=True)
    s.legend(L, 74, [("Python", SERIES[1]), ("NumPy", SERIES[2]), ("C", SERIES[0])])
    s.text(24, T + len(rungs) * 40 + 50, "Python rung measured at n = 96 (it would take minutes at 1024); all costs normalised per pair × dimension.", 11, T3)
    s.save("pdist_ladder.svg", "Pairwise distance implementations, from Python loops to AVX-512",
           "Horizontal log-scale bars from 42 ns for Python loops down to 0.05 ns for reordered C loops with AVX-512; "
           "NumPy broadcasting is slower than a NumPy row loop; the GEMM reformulation is the fastest NumPy rung.")


# ---------------------------------------------------------------- 6. measured roofline
def fig_roofline():
    peak = {r["lanes"]: r["gflops"] for r in E["peak_fma"]}
    lv = {r["n"]: r for r in E["saxpy_memory_levels"] if r["variant"] == "avx512"}
    bw = [("L1", lv[4096]["gbps"]), ("L2", lv[65536]["gbps"]), ("L3", lv[1 << 20]["gbps"]), ("DRAM", lv[1 << 26]["gbps"])]
    s = Svg(860, 500)
    header(s, "Measured roofline of one core: vectorization raises the flat roofs, never the slopes",
           f"GFLOP/s vs FLOP/byte (log-log) · roofs: register-only FMA chains · slopes: AVX-512 saxpy · {MACHINE}")
    L, R, T, B = 80, 700, 90, 440
    xlo, xhi, ylo, yhi = -2, 1.5, -0.5, 2.5

    def X(i):
        return L + (math.log10(i) - xlo) / (xhi - xlo) * (R - L)

    def Y(g):
        return B - (math.log10(g) - ylo) / (yhi - ylo) * (B - T)
    for v in (0.01, 0.1, 1, 10):
        s.line(X(v), T, X(v), B, GRID)
        s.text(X(v), B + 18, v, 11, T3, "middle")
    for g in (1, 10, 100):
        s.line(L, Y(g), R, Y(g), GRID)
        s.text(L - 8, Y(g) + 4, g, 11, T3, "end")
    s.text((L + R) / 2, B + 36, "arithmetic intensity (FLOP per byte)", 11, T3, "middle")
    s.line(L, B, R, B, AXIS)
    for name, b in bw:
        x0, x1 = 10 ** xlo, 10 ** xhi
        y0, y1 = b * x0, b * x1
        pts = []
        for xx in (x0, x1):
            yy = b * xx
            yy = min(max(yy, 10 ** ylo), 10 ** yhi)
            pts.append((X(yy / b), Y(yy)))
        s.line(pts[0][0], pts[0][1], pts[1][0], pts[1][1], "#5a5a5a", 1.5)
        ang = math.degrees(math.atan(((B - T) / (yhi - ylo)) / ((R - L) / (xhi - xlo))))
        gy = 10 ** (yhi - 0.12)
        lx = gy / b
        if 10 ** xlo < lx < 10 ** xhi:
            px, py = X(lx) - 70, Y(gy) + 70 * math.tan(math.radians(ang)) - 6
            s.text(px, py, f"{name} {b:.0f} GB/s", 10, T2, extra=f'transform="rotate(-{ang:.1f} {px:.1f} {py:.1f})"')
    names = {1: "scalar", 4: "SSE (4 lanes)", 8: "AVX2 (8 lanes)", 16: "AVX-512 (16 lanes)"}
    for i, lanes in enumerate((1, 4, 8, 16)):
        g = peak[lanes]
        ridge = g / bw[0][1]
        s.line(X(max(ridge, 10 ** xlo)), Y(g), R, Y(g), SERIES[i], 2)
        s.text(R + 8, Y(g) + 4, f"{names[lanes]} {g:.0f}", 11, T2)
    w = {r["kernel"] + "/" + r["variant"]: r for r in E["width_ladder_L1"]}
    acc = {r["kernel"] + "/" + r["variant"]: r for r in E["dot_accumulators_L1"]}
    pts = [
        ("saxpy, AVX-512, DRAM", 2 / 12, lv[1 << 26]["gflops"], SERIES[3], False),
        ("saxpy, AVX-512, L1", 2 / 12, lv[4096]["gflops"], SERIES[3], False),
        ("dot, strict IEEE, L1", 0.25, acc["dot/avx512"]["gflops"], SERIES[7], True),
        ("dot, 4 accumulators, L1", 0.25, acc["dot_avx512_4acc/avx512"]["gflops"], SERIES[0], False),
        ("leaky ReLU, AVX2 (not vectorized)", 1 / 8, w["leaky/avx2"]["gflops"], SERIES[7], True),
        ("leaky ReLU, AVX-512 masked", 1 / 8, w["leaky/avx512"]["gflops"], SERIES[0], False),
    ]
    for name, ai, g, col, hollow in pts:
        s.dot(X(ai), Y(g), col, 5, f"{name}: {g:.1f} GFLOP/s at {ai:.3f} FLOP/B", hollow=hollow)
    s.text(X(0.25) + 9, Y(acc["dot_avx512_4acc/avx512"]["gflops"]) + 4, "dot, 4 acc.", 11, T1)
    s.text(X(0.25) + 9, Y(acc["dot/avx512"]["gflops"]) + 4, "dot, strict IEEE (latency-bound)", 11, T1)
    s.text(X(2 / 12) - 9, Y(lv[1 << 26]["gflops"]) + 4, "saxpy from DRAM", 11, T1, "end")
    s.text(X(1 / 8) - 9, Y(w["leaky/avx2"]["gflops"]) + 4, "leaky ReLU, scalar code", 11, T1, "end")
    s.text(X(1 / 8) - 9, Y(w["leaky/avx512"]["gflops"]) + 4, "leaky ReLU, AVX-512 masked", 11, T1, "end")
    s.text(X(2 / 12) - 9, Y(lv[4096]["gflops"]) + 4, "saxpy from L1", 11, T1, "end")
    s.legend(L + 20, 72, [("near its roof", SERIES[0]), ("far below every roof", SERIES[7]), ("saxpy (defines the bandwidth lines)", SERIES[3])])
    s.save("roofline_measured.svg", "Measured roofline with vector-width compute roofs",
           "Four horizontal compute roofs from 13 to 186 GFLOP/s and four bandwidth diagonals; low-intensity kernels sit on "
           "the diagonals where wider registers cannot help; the strict-IEEE dot product sits far below every roof.")


# ---------------------------------------------------------------- 7. reduction orders
def fig_reduce():
    ro = E["reduction_orders"]
    order = [("sequential", "sequential (what the C source says)"), ("lanes_4", "4 lanes + tree (SSE-style)"),
             ("lanes_8", "8 lanes + tree (AVX2-style)"), ("lanes_16", "16 lanes + tree (AVX-512-style)"),
             ("pairwise", "pairwise tree"), ("numpy_sum", "numpy.sum (pairwise, blocked)"), ("kahan", "Kahan compensated")]
    s = Svg(860, 370)
    header(s, "Vectorizing a float sum changes its answer — usually for the better",
           f"1,000,000 float32 values ~ N(10⁴, 10³²); error vs the exact sum in ulps of the result (log scale)")
    L, R, T = 300, 760, 84
    lo, hi = math.log10(0.1), math.log10(300)
    for v in (0.1, 1, 10, 100):
        x = L + (math.log10(v) - lo) / (hi - lo) * (R - L)
        s.line(x, T - 6, x, T + len(order) * 36, GRID)
        s.text(x, T + len(order) * 36 + 16, v, 11, T3, "middle")
    s.text(R, T + len(order) * 36 + 32, "|error| in ulps", 11, T3, "end")
    for i, (k, name) in enumerate(order):
        e = ro["results"][k]["ulp_error"]
        y = T + i * 36 + 16
        s.text(L - 12, y + 4, name, 12, T1, "end")
        x = L + (math.log10(max(e, 0.1)) - lo) / (hi - lo) * (R - L)
        s.line(L, y, x, y, "#333333", 2)
        s.dot(x, y, SERIES[0] if k != "sequential" else SERIES[7], 5.5, f"{name}: {e:.1f} ulp, value {ro['results'][k]['value']:.0f}")
        s.text(x + 10, y + 4, f"{e:.1f}", 11, T2, mono=True)
    s.save("reduction_orders.svg", "Summation error by order", "Dot plot of summation error: the in-order sequential sum is furthest from "
           "the exact value, lane-split SIMD orders are closer, pairwise, numpy.sum and Kahan are within an ulp.")


# ---------------------------------------------------------------- 8. iteration space with dependences (diagram)
def fig_iteration_space():
    s = Svg(860, 430)
    header(s, "Seeing legality: an iteration space with its dependence arrows",
           "for i, j:  a[i][j] = a[i-1][j] + a[i][j-1]   — every point needs its left and lower neighbour")
    ox, oy, step, N = 120, 380, 34, 9
    for i in range(N):
        for j in range(N):
            x, y = ox + j * step, oy - i * step
            if j > 0:
                s.add(f'<line x1="{x - step + 6:.1f}" y1="{y:.1f}" x2="{x - 7:.1f}" y2="{y:.1f}" stroke="#4a4a4a" stroke-width="1.2" marker-end="url(#ah)"/>')
            if i > 0:
                s.add(f'<line x1="{x:.1f}" y1="{y + step - 6:.1f}" x2="{x:.1f}" y2="{y + 7:.1f}" stroke="#4a4a4a" stroke-width="1.2" marker-end="url(#ah)"/>')
    for i in range(N):
        for j in range(N):
            x, y = ox + j * step, oy - i * step
            w = i + j
            col = SERIES[w % 4] if w in (5, 6) else "#7a7a7a"
            s.dot(x, y, col, 4.5 if w in (5, 6) else 3.5, f"iteration (i={i}, j={j}), wavefront {w}")
    for w in (5, 6):
        x1, y1 = ox + 0, oy - w * step
        x2, y2 = ox + w * step, oy
        s.line(x1 - 14, y1 - 14, x2 + 14, y2 + 14, SERIES[w % 4], 1.5, "5,4")
        s.text(x1 - 18, y1 - 18 - (w - 5) * 0, f"i + j = {w}", 11, T2, "end")
    s.text(ox - 30, oy - (N - 1) * step / 2, "i", 14, T1, weight=600)
    s.text(ox + (N - 1) * step / 2, oy + 30, "j", 14, T1, weight=600)
    s.add('<defs><marker id="ah" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
          '<path d="M0,0 L6,3 L0,6 Z" fill="#4a4a4a"/></marker></defs>')
    tx = 470
    notes = [("Rows (fixed i, varying j)", "carry a dependence j-1 → j: not vectorizable along j."),
             ("Columns (fixed j, varying i)", "carry i-1 → i: not vectorizable along i either."),
             ("Anti-diagonals (i + j = const)", "have no arrows between their points: every point"),
             ("", "on a wavefront can run in the same vector instruction."),
             ("The skew", "re-index with t = i + j and vectorize the inner loop"),
             ("", "over the wavefront: same arithmetic, legal order.")]
    y = 120
    for head, body in notes:
        if head:
            y += 10
            s.text(tx, y, head, 13, T1, weight=600)
            y += 18
        s.text(tx, y, body, 12, T2)
        y += 18
    s.save("iteration_space.svg", "Iteration space of a 2-D recurrence with wavefronts",
           "A 9 by 9 grid of iterations with arrows to the right and upward; two anti-diagonal wavefronts are highlighted "
           "because points on them are mutually independent.")


# ---------------------------------------------------------------- 9. GEMM tiling, isometric (diagram)
def fig_gemm_iso():
    s = Svg(860, 470)
    header(s, "GEMM as a box of multiply-adds: tiling picks a sub-box that fits in fast memory",
           "C[M×N] += A[M×K] · B[K×N] — every point (m, n, k) is one FMA; a tile reuses its A and B panels many times")
    ox, oy = 330, 330
    u = 22
    ux, uy = (u, u / 2), (-u, u / 2)   # isometric unit vectors for n (right-down) and m (left-down)
    uz = (0, -u)                       # k goes up

    def P(m, n, k):
        return (ox + n * ux[0] + m * uy[0] + k * uz[0], oy + n * ux[1] + m * uy[1] + k * uz[1] - 90)

    def face(pts, fill, stroke="#5a5a5a", op=1.0, title=""):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"
        s.add(f'<path d="{d}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" stroke-width="1"><title>{escape(title)}</title></path>')

    M, N, K = 6, 8, 6
    face([P(0, 0, 0), P(0, N, 0), P(M, N, 0), P(M, 0, 0)], "#161616", title="the full M×N×K iteration box (bottom)")
    face([P(0, 0, 0), P(0, 0, K), P(M, 0, K), P(M, 0, 0)], "#161616")
    face([P(0, 0, 0), P(0, N, 0), P(0, N, K), P(0, 0, K)], "#161616")
    # the tile: m 2..4, n 3..6, k 1..4
    m0, m1, n0, n1, k0, k1 = 2, 4, 3, 6, 1, 4
    face([P(m0, n0, k1), P(m0, n1, k1), P(m1, n1, k1), P(m1, n0, k1)], SERIES[0], "#9ec5f4", 0.85, "one tile: Mt×Nt×Kt FMAs")
    face([P(m1, n0, k0), P(m1, n1, k0), P(m1, n1, k1), P(m1, n0, k1)], SERIES[0], "#9ec5f4", 0.55)
    face([P(m0, n1, k0), P(m1, n1, k0), P(m1, n1, k1), P(m0, n1, k1)], SERIES[0], "#9ec5f4", 0.4)
    # the A panel (m x k) projected on the n=0 wall, B panel (k x n) on the m=0 wall, C tile on the floor
    face([P(m0, 0, k0), P(m1, 0, k0), P(m1, 0, k1), P(m0, 0, k1)], SERIES[1], "#f0a07a", 0.8, "A panel (Mt×Kt) — reused for every n in the tile")
    face([P(0, n0, k0), P(0, n1, k0), P(0, n1, k1), P(0, n0, k1)], SERIES[2], "#7fd8b4", 0.8, "B panel (Kt×Nt) — reused for every m in the tile")
    face([P(m0, n0, 0), P(m0, n1, 0), P(m1, n1, 0), P(m1, n0, 0)], SERIES[3], "#f3cf7a", 0.8, "C tile (Mt×Nt) — accumulates over k, stays in registers")
    for (a, b, c), lab in (((M + 0.4, 0, 0), "m"), ((0, N + 0.4, 0), "n"), ((0, 0, K + 0.5), "k")):
        x, y = P(a, b, c)
        s.text(x, y, lab, 14, T1, "middle", 600)
    tx = 560
    rows = [(SERIES[0], "Tile", "Mt·Nt·Kt FMAs, done in fast memory"), (SERIES[1], "A panel", "Mt·Kt values, read once per tile"),
            (SERIES[2], "B panel", "Kt·Nt values, read once per tile"), (SERIES[3], "C tile", "Mt·Nt accumulators in registers")]
    y = 140
    for col, head, body in rows:
        s.add(f'<rect x="{tx}" y="{y - 10}" width="12" height="12" rx="2" fill="{col}"/>')
        s.text(tx + 20, y, head, 13, T1, weight=600)
        s.text(tx + 20, y + 18, body, 12, T2)
        y += 48
    s.text(tx, y + 6, "Reuse = FMAs / values loaded", 12, T1, weight=600)
    s.text(tx, y + 26, "= Mt·Nt·Kt / (Mt·Kt + Kt·Nt)", 12, T2, mono=True)
    s.text(tx, y + 46, "= 1 / (1/Nt + 1/Mt)  per k-step", 12, T2, mono=True)
    s.text(tx, y + 70, "Bigger tiles → more FLOPs per byte →", 12, T2)
    s.text(tx, y + 88, "the kernel climbs the roofline slope.", 12, T2)
    s.save("gemm_tiles.svg", "GEMM iteration box with one tile and its operand panels",
           "Isometric box of M by N by K multiply-adds; one tile is highlighted with its A panel, B panel and C tile "
           "projected onto the three walls.")


if __name__ == "__main__":
    fig_saxpy_levels()
    fig_width_heatmap()
    fig_stride()
    fig_dot()
    fig_ladder()
    fig_roofline()
    fig_reduce()
    fig_iteration_space()
    fig_gemm_iso()
