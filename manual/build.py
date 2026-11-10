"""
Build the DOMINION manual PDF.

  1. draws the logo (a small tessellation: seeds and their domains)
  2. runs capture.py if an image is given (screenshots from the real app)
  3. prints dominion.html twice with headless Edge: the first pass finds the
     page each chapter starts on, the second prints the contents with them

Run:  python build.py                          (reuse the screenshots in img\\)
      python build.py C:\\path\\to\\image.tif    (capture again first; needs the
                                               Full dominion environment)

Needs pymupdf (pip install pymupdf) and Microsoft Edge.
"""
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT_NAME = "Dominion_Manual.pdf"


def write_logo():
    """Seven seeds in a disc and the domains they claim: each point of the
    disc belongs to its nearest seed (a Voronoi diagram, clipped to the
    circle by sampling), drawn as cell outlines with a dot at each seed."""
    seeds = [(0.0, 0.0)] + [(0.58 * math.cos(a), 0.58 * math.sin(a))
                            for a in [i * math.pi / 3 + 0.35 for i in range(6)]]
    jitter = [(0, 0), (.05, -.03), (-.04, .06), (.06, .05), (-.05, -.04), (.03, .07), (-.06, .02)]
    seeds = [(x + j[0], y + j[1]) for (x, y), j in zip(seeds, jitter)]
    n = 220
    owner = {}
    for i in range(n):
        for j in range(n):
            x, y = -1 + 2 * (j + .5) / n, -1 + 2 * (i + .5) / n
            if x * x + y * y <= 1:
                owner[i, j] = min(range(len(seeds)), key=lambda k: (x - seeds[k][0]) ** 2 + (y - seeds[k][1]) ** 2)
    # border segments between pixels of different owners
    segs = []
    s = 100 / n
    for (i, j), k in owner.items():
        if owner.get((i, j + 1), k) != k and (i, j + 1) in owner:
            segs.append(f"M{(j + 1) * s:.1f} {i * s:.1f}v{s:.2f}")
        if owner.get((i + 1, j), k) != k and (i + 1, j) in owner:
            segs.append(f"M{j * s:.1f} {(i + 1) * s:.1f}h{s:.2f}")
    dots = "".join(f'<circle cx="{50 + 50 * x:.1f}" cy="{50 + 50 * y:.1f}" r="4.2"/>' for x, y in seeds)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="-2 -2 104 104">
<circle cx="50" cy="50" r="49" fill="#e8f2f1" stroke="#0f6e6a" stroke-width="2.2"/>
<path d="{"".join(segs)}" stroke="#0f6e6a" stroke-width="1.6" fill="none" stroke-linecap="round"/>
<g fill="#1d2326">{dots}</g>
</svg>'''
    with open(os.path.join(IMG, "logo.svg"), "w", encoding="utf-8") as f:
        f.write(svg)


def edge_print(html_path, pdf_path):
    """msedge.exe returns before the PDF is written, so the old file is
    removed first and the new one waited for until its size settles."""
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    url = "file:///" + html_path.replace("\\", "/")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=8000", f"--print-to-pdf={pdf_path}", url],
                   check=True, capture_output=True)
    last = -1
    for _ in range(240):
        time.sleep(0.5)
        size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else -1
        if size > 0 and size == last:
            return
        last = size
    raise SystemExit(f"Edge did not write {pdf_path}")


def print_pdf():
    """Two passes: markers find each chapter's page, then the contents get
    their numbers."""
    import pymupdf
    with open(os.path.join(HERE, "dominion.html"), encoding="utf-8") as f:
        src = f.read()
    ids = re.findall(r'<section class="chapter[^"]*" id="([\w-]+)"', src)
    tmp_html = os.path.join(HERE, "_print.html")
    tmp_pdf = os.path.join(HERE, "_pass1.pdf")
    out = os.path.join(HERE, OUT_NAME)

    marked = src
    for i in ids:
        marked = marked.replace(f'id="{i}">', f'id="{i}"><span class="mark">@@{i}@@</span>', 1)
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(marked)
    edge_print(tmp_html, tmp_pdf)
    pages = {}
    with pymupdf.open(tmp_pdf) as doc:
        for n, pg in enumerate(doc):
            for m in re.findall(r"@@([\w-]+)@@", pg.get_text()):
                pages.setdefault(m, n + 1)
    missing = [i for i in ids if i not in pages]
    if missing:
        raise SystemExit(f"chapter markers not found: {missing}")

    final = re.sub(r'<span class="pg" data-ref="([\w-]+)">[^<]*</span>',
                   lambda m: f'<span class="pg">{pages[m.group(1)]}</span>', src)
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(final)
    edge_print(tmp_html, out)
    os.remove(tmp_html)
    os.remove(tmp_pdf)
    with pymupdf.open(out) as doc:
        print(f"wrote {out} ({doc.page_count} pages, {os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    os.makedirs(IMG, exist_ok=True)
    write_logo()
    if len(sys.argv) > 1:
        subprocess.run([sys.executable, os.path.join(HERE, "capture.py"), *sys.argv[1:]], check=True)
    print_pdf()


if __name__ == "__main__":
    main()
