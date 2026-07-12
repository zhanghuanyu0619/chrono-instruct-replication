#!/usr/bin/env python3
"""Combined sweep figure — pure stdlib, emits a self-contained SVG (and an
optional HTML wrapper for viewing). No matplotlib/numpy dependency, so it runs
anywhere Python does.

Panels:
  Row 1 — one panel per SFT stage: validation loss vs optimizer step, one line
          per vintage.
  Row 2 — headline: best validation loss per stage across vintages, direct-
          labeled, showing the monotone improvement with a later cutoff.

Color is SEQUENTIAL (vintage year is ordered): a single blue hue light->dark,
1999 lightest -> 2024 darkest, so "more recent cutoff" reads as "more ink".
Palette steps from the dataviz skill's ordinal blue ramp.

Usage: python scripts/plot_sweep_combined.py [results_dir] [out.svg] [out.html]
"""
import csv
import json
import os
import sys

# Resolve paths against the repo root (parent of scripts/), never the caller's CWD,
# so the figure always lands under results/, not wherever the script was invoked.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "results")
# Combined artifacts live in results/combined/ (a subfolder), not the results root.
OUT_SVG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RESULTS, "combined", "sweep_combined.svg")
OUT_HTML = sys.argv[3] if len(sys.argv) > 3 else None

YEARS = [1999, 2005, 2010, 2015, 2020, 2024]
YEAR_COLOR = {
    1999: "#86b6ef", 2005: "#5598e7", 2010: "#3987e5",
    2015: "#256abf", 2020: "#1c5cab", 2024: "#0d366b",
}
STAGES = ["stage1_scratch", "stage2_self_instruct", "stage3_tulu"]
STAGE_TITLE = {
    "stage1_scratch": "Stage 1 — scratch",
    "stage2_self_instruct": "Stage 2 — self-instruct",
    "stage3_tulu": "Stage 3 — Tulu",
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e6e2", "#fcfcfb"


def read_val_series(run_dir):
    per = {}
    with open(os.path.join(run_dir, "metrics.csv")) as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            if row["loss"] in ("", "nan"):
                continue
            per.setdefault(row["stage"], {})[int(row["step"])] = float(row["loss"])
    return {s: (sorted(d), [d[k] for k in sorted(d)]) for s, d in per.items()}


def final_val(run_dir):
    with open(os.path.join(run_dir, "summary.json")) as f:
        return json.load(f)["final_val_loss"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Panel:
    """A plot area mapping data coords -> svg pixels, with axes/grid helpers."""
    def __init__(self, x, y, w, h, xlim, ylim):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim

    def px(self, xv):
        if self.x1 == self.x0:
            return self.x + self.w / 2
        return self.x + (xv - self.x0) / (self.x1 - self.x0) * self.w

    def py(self, yv):
        return self.y + self.h - (yv - self.y0) / (self.y1 - self.y0) * self.h


def nice_ticks(lo, hi, n=4):
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 0.01
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if (hi - lo) / step <= n + 1:
            break
    t, out = (int(lo / step) * step), []
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            out.append(round(t, 4))
        t += step
    return out


def build_svg():
    data = {y: read_val_series(os.path.join(RESULTS, f"chrono-instruct-{y}")) for y in YEARS}
    finals = {y: final_val(os.path.join(RESULTS, f"chrono-instruct-{y}")) for y in YEARS}

    W, H = 1360, 880
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

    # Titles
    s.append(f'<text x="24" y="40" font-size="22" font-weight="700" fill="{INK}">'
             'ChronoGPT-Instruct SFT replication — validation loss across the vintage sweep</text>')
    s.append(f'<text x="24" y="65" font-size="14" fill="{MUTED}">'
             '6 chronologically-consistent vintages · 3-stage curriculum · seed 123 · block 1792 · '
             'all runs converged, none failed</text>')

    # Legend — horizontal strip under the subtitle (keeps the right margin clear)
    lx0, ly = 24, 92
    s.append(f'<text x="{lx0}" y="{ly+4}" font-size="12.5" font-weight="600" fill="{MUTED}">cutoff year:</text>')
    cx = lx0 + 92
    for yr in YEARS:
        s.append(f'<line x1="{cx}" y1="{ly}" x2="{cx+22}" y2="{ly}" stroke="{YEAR_COLOR[yr]}" stroke-width="3"/>')
        s.append(f'<circle cx="{cx+11}" cy="{ly}" r="3.4" fill="{YEAR_COLOR[yr]}"/>')
        s.append(f'<text x="{cx+29}" y="{ly+4}" font-size="12.5" fill="{INK}">{yr}</text>')
        cx += 92

    # ---- Row 1: 3 stage panels ----
    top, ph = 138, 244
    left, right_pad, gap = 60, 40, 46
    avail = W - left - right_pad
    pw = (avail - 2 * gap) / 3
    for ci, stage in enumerate(STAGES):
        allv = [v for y in YEARS if stage in data[y] for v in data[y][stage][1]]
        allx = [v for y in YEARS if stage in data[y] for v in data[y][stage][0]]
        ylo, yhi = min(allv), max(allv)
        pad = (yhi - ylo) * 0.08 or 0.05
        xlo, xhi = min(allx), max(allx)
        P = Panel(left + ci * (pw + gap), top, pw, ph, (xlo, xhi), (ylo - pad, yhi + pad))
        # grid + y ticks
        for tv in nice_ticks(ylo - pad, yhi + pad, 4):
            gy = P.py(tv)
            s.append(f'<line x1="{P.x:.1f}" y1="{gy:.1f}" x2="{P.x+P.w:.1f}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="0.8"/>')
            s.append(f'<text x="{P.x-8:.1f}" y="{gy+4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{tv:.2f}</text>')
        # x ticks
        for tv in nice_ticks(xlo, xhi, 3):
            gx = P.px(tv)
            lbl = f'{int(tv/1000)}k' if tv >= 1000 else str(int(tv))
            s.append(f'<text x="{gx:.1f}" y="{P.y+P.h+18:.1f}" font-size="11" text-anchor="middle" fill="{MUTED}">{lbl}</text>')
        # lines
        for yr in YEARS:
            if stage not in data[yr]:
                continue
            xs, ys = data[yr][stage]
            pts = " ".join(f'{P.px(a):.1f},{P.py(b):.1f}' for a, b in zip(xs, ys))
            s.append(f'<polyline points="{pts}" fill="none" stroke="{YEAR_COLOR[yr]}" stroke-width="2" '
                     'stroke-linejoin="round" stroke-linecap="round"/>')
            # end marker
            s.append(f'<circle cx="{P.px(xs[-1]):.1f}" cy="{P.py(ys[-1]):.1f}" r="3" fill="{YEAR_COLOR[yr]}"/>')
        s.append(f'<text x="{P.x+P.w/2:.1f}" y="{P.y-12:.1f}" font-size="14" font-weight="600" '
                 f'text-anchor="middle" fill="{INK}">{esc(STAGE_TITLE[stage])}</text>')
        s.append(f'<text x="{P.x+P.w/2:.1f}" y="{P.y+P.h+36:.1f}" font-size="11.5" '
                 f'text-anchor="middle" fill="{MUTED}">optimizer step</text>')
    s.append(f'<text x="18" y="{top+ph/2:.1f}" font-size="12" fill="{MUTED}" '
             f'transform="rotate(-90 18 {top+ph/2:.1f})" text-anchor="middle">validation loss</text>')

    # ---- Row 2: headline panel ----
    b_top = top + ph + 116
    bh = 210
    b_width = avail - 130   # reserve right margin for the direct year labels
    BP = Panel(left, b_top, b_width, bh, (0, len(STAGES) - 1), (0.72, 1.50))
    s.append(f'<text x="{left}" y="{b_top-24:.1f}" font-size="16" font-weight="700" fill="{INK}">'
             'Best validation loss falls monotonically as the knowledge cutoff moves forward</text>')
    for tv in nice_ticks(0.72, 1.50, 5):
        gy = BP.py(tv)
        s.append(f'<line x1="{BP.x:.1f}" y1="{gy:.1f}" x2="{BP.x+BP.w:.1f}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="0.8"/>')
        s.append(f'<text x="{BP.x-8:.1f}" y="{gy+4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{tv:.2f}</text>')
    for i, stage in enumerate(STAGES):
        gx = BP.px(i)
        anchor = "end" if i == len(STAGES) - 1 else ("start" if i == 0 else "middle")
        s.append(f'<text x="{gx:.1f}" y="{BP.y+BP.h+24:.1f}" font-size="12.5" font-weight="600" '
                 f'text-anchor="{anchor}" fill="{INK}">{esc(STAGE_TITLE[stage])}</text>')
    ends = {}
    for yr in YEARS:
        ys = [finals[yr][st] for st in STAGES]
        ends[yr] = ys[-1]
        pts = " ".join(f'{BP.px(i):.1f},{BP.py(v):.1f}' for i, v in enumerate(ys))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{YEAR_COLOR[yr]}" stroke-width="2.4" '
                 'stroke-linejoin="round"/>')
        for i, v in enumerate(ys):
            s.append(f'<circle cx="{BP.px(i):.1f}" cy="{BP.py(v):.1f}" r="4.5" fill="{YEAR_COLOR[yr]}"/>')
    # Right-side label column: the six final values (0.785–0.869) are too close to
    # label in place, so spread them evenly (year order == value order, monotone)
    # with a short leader from each line's endpoint.
    xe = BP.px(len(STAGES) - 1)
    lab_x = xe + 30
    y0 = BP.py(ends[YEARS[0]]) - 8
    for i, yr in enumerate(YEARS):
        ly_ = y0 + i * 20
        ey = BP.py(ends[yr])
        s.append(f'<path d="M{xe+5:.1f},{ey:.1f} L{lab_x-6:.1f},{ly_-4:.1f}" fill="none" '
                 f'stroke="{YEAR_COLOR[yr]}" stroke-width="1" opacity="0.55"/>')
        s.append(f'<text x="{lab_x:.1f}" y="{ly_:.1f}" font-size="12.5" font-weight="700" '
                 f'fill="{YEAR_COLOR[yr]}">{yr}: {ends[yr]:.3f}</text>')
    s.append(f'<text x="18" y="{b_top+bh/2:.1f}" font-size="12" fill="{MUTED}" '
             f'transform="rotate(-90 18 {b_top+bh/2:.1f})" text-anchor="middle">best validation loss</text>')

    s.append('</svg>')
    return "\n".join(s)


def main():
    svg = build_svg()
    os.makedirs(os.path.dirname(OUT_SVG) or ".", exist_ok=True)
    with open(OUT_SVG, "w") as f:
        f.write(svg)
    print(f"wrote {OUT_SVG}")
    if OUT_HTML:
        with open(OUT_HTML, "w") as f:
            f.write(f'<div style="max-width:100%;overflow-x:auto">{svg}</div>')
        print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
