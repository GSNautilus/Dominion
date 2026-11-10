"""
Screen captures for the DOMINION manual.

Drives the real DOMINION dock in a real napari window, exactly as a user
would (load settings, click Run, toggle sections), and saves:

  img/win_*.png     the whole napari window (layer list, canvas, dock)
  img/view_*.jpg    the canvas alone, zoomed into one patch of tissue
  img/panel_*.png   one dock section, as it looks after its Run
  img/sholl_*.svg   one cell's skeleton, rings and Sholl profile

Nothing from the image itself is written except these pictures: no CSVs,
no label arrays. (Measurements export goes to a temp folder and is read
only to check the column list.)

Run in the Full `dominion` environment (StarDist is needed for the
nuclei-mode shots), with any 2-channel CYX TIFF (signal, nuclei) that has
its pixel size in the TIFF metadata:

  python capture.py C:\\path\\to\\image.tif            (every shot)
  python capture.py C:\\path\\to\\image.tif nuclei     (only shot groups
                                                        whose name matches)

A napari window opens on screen while it runs (about two minutes).
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")

# The parameters the manual walks through. Tuned on a GFAP + DAPI mouse
# cortex section imaged at 0.29 µm/px; see the "Choosing parameters" chapter.
SETTINGS = {
    "signal_seed_finding": {
        "method": "Local maxima of smoothed signal",
        "sigma_um": 6.6,
        "signal_threshold": 266.0,
        "min_distance_um": 15.0,
        "peak_strength": 338.811,
    },
    "tessellation": {
        "signal_influence": 0.8,
        "smoothing_sigma_um": 0.1,
        "min_signal": 315.0,
        "min_area_um2": 0.0,
        "max_area_um2": 0.0,
    },
    "skeletons": {"min_branch_length_um": 0.0, "min_branch_signal": 0.0, "force_tree": True},
    "sholl_analysis": {"ring_spacing_um": 5.0},
    "measurements": {"channels_checked": ["signal", "nuclei"]},
}

WIN = (1640, 1000)       # napari window, px
DOCK_W = 440             # DOMINION dock width, px
PANEL_W = 420            # width of the single-section grabs

import napari  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication, QPushButton, QScrollArea  # noqa: E402

import dominion  # noqa: E402
from dominion.app import build_dock_widget  # noqa: E402
from dominion.io import load_image  # noqa: E402
from dominion.state import AppState  # noqa: E402
from dominion.widgets.common import CollapsibleSection  # noqa: E402


# ---------------------------------------------------------------- helpers
def pump(seconds=0.25):
    end = time.time() + seconds
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


class App:
    """One napari window with the DOMINION dock, in signal or nuclei mode."""

    def __init__(self, path, mode):
        self.state = AppState()
        self.img = load_image(path)
        self.state.set("image", self.img)
        self.viewer = napari.Viewer(title=f"DOMINION ({mode})", show=True)
        self.dock = build_dock_widget(self.state, self.viewer, mode=mode)
        self.dw = self.viewer.window.add_dock_widget(self.dock, name=f"DOMINION ({mode})", area="right")
        self.dw.setMinimumWidth(DOCK_W)
        qw = self.viewer.window._qt_window
        qw.resize(*WIN)
        qw.move(40, 40)
        pump(1.5)
        self.px = float(self.img.pixel_size_um)

    # sections and buttons, found by their on-screen text
    def section(self, title):
        for s in self.dock.findChildren(CollapsibleSection):
            if s._toggle.text() == title:
                return s
        raise KeyError(title)

    def sections(self):
        return self.dock.findChildren(CollapsibleSection)

    def only_open(self, *titles):
        for s in self.sections():
            want = s._toggle.text() in titles
            if s._toggle.isChecked() != want:
                s._toggle.click()
        pump(0.3)
        if titles:   # bring the first open section into view in the dock
            area = self.dw.findChild(QScrollArea)
            if area is not None:
                area.ensureWidgetVisible(self.section(titles[0]), 0, 0)
        pump(0.3)

    def click(self, text):
        for b in self.dock.findChildren(QPushButton):
            if b.text() == text:
                b.click()
                pump(0.4)
                return
        raise KeyError(text)

    def set_slider(self, label, value):
        """Move a slider like a hand would (its live handlers fire)."""
        from dominion.widgets.common import NumericSlider
        for sl in self.dock.findChildren(NumericSlider):
            if sl._label.text().startswith(label):
                sl.set_value(value)
                pump(0.4)
                return
        raise KeyError(label)

    def apply(self, settings):
        self.state.apply_all_settings(settings)
        pump(0.3)

    def layer(self, name):
        return self.viewer.layers[name]

    def show_only(self, *names):
        for lay in self.viewer.layers:
            lay.visible = lay.name in names

    # camera, in microns
    def look(self, center_rc_px, width_um):
        cy, cx = center_rc_px
        self.viewer.camera.center = (cy * self.px, cx * self.px)
        canvas_w = self.viewer.window._qt_viewer.canvas.size[0]
        self.viewer.camera.zoom = canvas_w / width_um
        pump(0.5)

    def home(self):
        self.viewer.reset_view()
        pump(0.5)

    # saving
    def save_window(self, name):
        arr = self.viewer.screenshot(canvas_only=False, flash=False)
        Image.fromarray(arr).convert("RGB").save(os.path.join(OUT, f"win_{name}.png"), optimize=True)
        print("  win", name)

    def save_view(self, name):
        arr = self.viewer.screenshot(canvas_only=True, flash=False)
        Image.fromarray(arr).convert("RGB").save(os.path.join(OUT, f"view_{name}.jpg"), quality=90)
        print("  view", name)

    def save_panel(self, title, name):
        s = self.section(title)
        self.only_open(title)
        old = s.width()
        s.setFixedWidth(PANEL_W)
        pump(0.3)
        s.grab().save(os.path.join(OUT, f"panel_{name}.png"))
        s.setMinimumWidth(0)
        s.setMaximumWidth(16777215)
        s.resize(old, s.height())
        pump(0.2)
        print("  panel", name)

    def close(self):
        self.viewer.close()
        pump(0.5)


def busiest_point(points_rc, px, window_um=260.0):
    """The seed whose neighbourhood holds the most seeds: a full-looking
    patch of tissue for the zoomed views."""
    p = np.asarray(points_rc, dtype=float) * px
    best, best_n = 0, -1
    for i, q in enumerate(p):
        n = int(np.sum(np.all(np.abs(p - q) < window_um / 2, axis=1)))
        if n > best_n:
            best, best_n = i, n
    return np.asarray(points_rc[best], dtype=float)


# ------------------------------------------------------------ signal mode
def shots_signal(path):
    a = App(path, "signal")
    st = a.state

    # 1. As it opens: image loaded, nothing run.
    a.only_open("Signal seed-finding")
    a.save_window("start")
    a.save_panel("Channels", "channels")
    a.save_panel("Batch", "batch")

    # 2. Seeds.
    a.apply(SETTINGS)
    a.click("Run seed-finding")
    a.set_slider("Min peak", SETTINGS["signal_seed_finding"]["peak_strength"])   # the live filter
    seeds_rc = st.nuclei.centroids
    focus = busiest_point(seeds_rc, a.px)
    a.save_panel("Signal seed-finding", "signal_seeds")
    a.only_open("Signal seed-finding")
    a.look(focus, 520)
    a.save_window("seeds")

    # 3. Tessellation, with two contrasts the manual explains.
    def tess(**kw):
        s = dict(SETTINGS["tessellation"], **kw)
        a.apply({"tessellation": s})
        a.click("Run tessellation")

    a.show_only("Signal", "Domains", "Object seeds")
    for name, kw in (("si0", {"signal_influence": 0.0, "min_signal": 0.0}),
                     ("minsig0", {"min_signal": 0.0})):
        tess(**kw)
        a.layer("Domains").opacity = 0.45
        a.look(focus, 300)
        a.save_view(f"tess_{name}")
    tess()
    a.layer("Domains").opacity = 0.45
    a.look(focus, 300)
    a.save_view("tess_main")
    a.save_panel("Tessellation", "tessellation")

    a.only_open("Tessellation")
    a.show_only("Signal", "Domains")
    a.layer("Domains").opacity = 0.55
    a.home()
    a.save_window("domains")

    # 4. Skeletons + Sholl.
    a.click("Run skeletonization")
    a.save_panel("Skeletons", "skeletons")
    a.show_only("Signal", "Domains", "Skeletons")
    a.layer("Domains").opacity = 0.18
    a.look(focus, 300)
    a.save_view("skeletons")
    a.click("Run Sholl analysis")
    a.save_panel("Sholl analysis", "sholl")
    sholl_figure(a, focus)

    # 5. Measurements (export to a temp folder, only to list the columns).
    a.click("Run measurements")
    a.save_panel("Measurements", "measurements")
    export_columns(a)

    # 6. ROI: a rectangle around the focus patch, then seeds + tessellation again.
    a.home()
    roi = a.layer("ROI")
    cy, cx = focus                       # the ROI layer's data are in pixels
    h = 420.0 / a.px
    roi.add_rectangles(np.array([[cy - h / 2, cx - h * 0.7], [cy + h / 2, cx + h * 0.7]]))
    a.click("Apply ROI")
    a.apply(SETTINGS)          # a new tissue mask resets the sliders, as a new image does
    a.click("Run seed-finding")
    a.set_slider("Min peak", SETTINGS["signal_seed_finding"]["peak_strength"])
    a.click("Run tessellation")
    a.save_panel("ROI", "roi")
    a.show_only("Signal", "ROI", "Domains", "Object seeds")
    a.layer("Domains").opacity = 0.5
    a.only_open("ROI")
    a.look(focus, 760)
    a.save_window("roi")
    a.close()


def sholl_figure(a, focus):
    """One cell near the focus: signal, domain outline, skeleton, Sholl
    rings (img/sholl_cell.png) and its intersection profile (sholl_profile.svg)."""
    st = a.state
    labels = st.tessellation.domain_labels
    # a well-branched cell near the focus whose domain is one piece
    from scipy import ndimage as ndi
    win = int(160 / a.px)
    y0w, x0w = max(int(focus[0]) - win, 0), max(int(focus[1]) - win, 0)
    sub = labels[y0w:int(focus[0]) + win, x0w:int(focus[1]) + win]
    best, best_score = None, -1
    for d in set(np.unique(sub)) - {0}:
        info = st.sholl.per_domain.get(int(d))
        if info is None:
            continue
        m = sub == d
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        frac = np.bincount(lab.ravel())[1:].max() / m.sum() if n else 0
        score = info["peak_intersections"] * (frac > 0.97)
        if score > best_score:
            best, best_score = int(d), score
    did = best
    info = st.sholl.per_domain[did]
    # root: the seed inside this domain
    seeds = st.nuclei.centroids[st.seeds.kept_indices]
    inside = [s for s in seeds if labels[int(s[0]), int(s[1])] == did]
    root = np.asarray(inside[0] if inside else np.argwhere(labels == did).mean(0))

    r_um = float(info["max_radius_um"]) + 6
    r_px = int(r_um / a.px)
    y0, x0 = int(root[0]) - r_px, int(root[1]) - r_px
    y1, x1 = int(root[0]) + r_px, int(root[1]) + r_px
    H, W = labels.shape
    ys, xs = slice(max(y0, 0), min(y1, H)), slice(max(x0, 0), min(x1, W))
    sig = a.img.signal[ys, xs].astype(np.float32)
    lo, hi = np.percentile(a.img.signal[a.img.tissue_mask], (1, 99.7))
    g = np.clip((sig - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([g * 0.15, g * 0.95, g * 0.35], -1)
    dom = labels[ys, xs] == did
    rgb[~dom] *= 0.35
    rgb[dom] = rgb[dom] * 0.8 + np.array([0.10, 0.10, 0.14])
    sk = st.skeletons.skeleton_label_image[ys, xs] == did
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    scale = 900 / max(im.size)
    size = (int(im.size[0] * scale), int(im.size[1] * scale))
    im = im.resize(size, Image.LANCZOS)
    skm = Image.fromarray(sk.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    im.paste(Image.new("RGB", size, (255, 70, 215)), (0, 0), skm)
    d = ImageDraw.Draw(im)
    ry, rx = (root[0] - ys.start) * scale, (root[1] - xs.start) * scale
    for r in info["radii_um"]:
        rr = r / a.px * scale
        d.ellipse((rx - rr, ry - rr, rx + rr, ry + rr), outline=(255, 214, 90), width=2)
    d.ellipse((rx - 9, ry - 9, rx + 9, ry + 9), fill=(255, 255, 0))
    bar_px = 20 / a.px * scale   # 20 µm scale bar
    d.rectangle((im.size[0] - 30 - bar_px, im.size[1] - 34, im.size[0] - 30, im.size[1] - 26), fill=(255, 255, 255))
    im.save(os.path.join(OUT, "sholl_cell.png"), optimize=True)

    radii = [float(x) for x in info["radii_um"]]
    counts = [int(x) for x in info["intersections"]]
    meta = {k: float(info[k]) for k in ("peak_intersections", "peak_radius_um", "max_radius_um",
                                        "critical_radius_um", "auc", "ramification_index")}
    write_profile_svg(radii, counts, meta, float(st.sholl.params["ring_spacing_um"]))
    print("  sholl figure, domain", did)


def write_profile_svg(radii, counts, meta, spacing):
    W, H, L, R, T, B = 560, 300, 52, 16, 18, 46
    xmax = max(radii) + spacing
    ymax = max(max(counts) + 1, 4)
    sx = lambda x: L + (W - L - R) * x / xmax  # noqa: E731
    sy = lambda y: H - B - (H - T - B) * y / ymax  # noqa: E731
    out = []
    for yv in range(0, ymax + 1, max(1, ymax // 5)):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{sy(yv) + 4:.1f}" class="tick" text-anchor="end">{yv}</text>')
    step = 10 if xmax > 40 else 5
    for xv in range(0, int(xmax) + 1, step):
        out.append(f'<text x="{sx(xv):.1f}" y="{H - B + 17}" class="tick" text-anchor="middle">{xv}</text>')
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(radii, counts))
    out.append(f'<polyline points="{pts}" class="line"/>')
    for x, y in zip(radii, counts):
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" class="dot"/>')
    pr, cr = meta["peak_radius_um"], meta["critical_radius_um"]
    out.append(f'<line x1="{sx(pr):.1f}" x2="{sx(pr):.1f}" y1="{T}" y2="{H - B}" class="mark"/>')
    out.append(f'<text x="{sx(pr) + 5:.1f}" y="{T + 10}" class="lab">peak {int(meta["peak_intersections"])} at {pr:.0f} µm</text>')
    if cr > 0:
        out.append(f'<line x1="{sx(cr):.1f}" x2="{sx(cr):.1f}" y1="{T + 18}" y2="{H - B}" class="mark2"/>')
        out.append(f'<text x="{sx(cr) + 5:.1f}" y="{T + 28}" class="lab2">critical radius {cr:.0f} µm</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" class="axis"/>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" class="title" text-anchor="middle">distance from seed (µm)</text>')
    out.append(f'<text x="14" y="{(T + H - B) / 2}" class="title" text-anchor="middle" '
               f'transform="rotate(-90 14 {(T + H - B) / 2})">intersections</text>')
    style = """text { font-family: "Segoe UI", sans-serif; fill: #1d2326; }
.tick { font-size: 11px; fill: #5d666b; } .title { font-size: 12px; fill: #3a4246; }
.lab { font-size: 11px; fill: #0f6e6a; font-weight: 600; } .lab2 { font-size: 11px; fill: #8a3b8f; font-weight: 600; }
.grid { stroke: #e3e6e4; stroke-width: 1; } .axis { stroke: #1d2326; stroke-width: 1.2; }
.line { fill: none; stroke: #1d2326; stroke-width: 1.6; } .dot { fill: #1d2326; }
.mark { stroke: #0f6e6a; stroke-width: 1.2; stroke-dasharray: 4 3; }
.mark2 { stroke: #8a3b8f; stroke-width: 1.2; stroke-dasharray: 4 3; }"""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">\n<style>{style}</style>\n'
           + "\n".join(out) + "\n</svg>")
    open(os.path.join(OUT, "sholl_profile.svg"), "w", encoding="utf-8").write(svg)


def export_columns(a):
    """Write the CSVs to a temp folder, check the column list the manual
    documents is the one the code writes, then delete them."""
    from dominion import submenu_measurements as sm
    st = a.state
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x_domains.csv")
        from pathlib import Path
        sm._write_domains_csv(Path(p), st.measurements, a.px, skeletons=st.skeletons, sholl=st.sholl)
        sm._write_summary_csv(Path(tmp) / "x_domains_summary.csv", st.measurements, st.image,
                              st.tessellation, st.seeds, st.nuclei, skeletons=st.skeletons, sholl=st.sholl)
        cols = open(p, encoding="utf-8").readline().strip().split(",")
        summ = [ln.split(",")[0] for ln in open(os.path.join(tmp, "x_domains_summary.csv"), encoding="utf-8")]
    with open(os.path.join(OUT, "csv_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"domains": cols, "summary": summ}, f, indent=1)
    print("  csv columns:", len(cols), "per-domain,", len(summ), "summary rows")


# ------------------------------------------------------------ nuclei mode
def shots_nuclei(path):
    a = App(path, "nuclei")
    st = a.state
    a.only_open("Nuclei segmentation")
    a.click("Run StarDist")
    t0 = time.time()
    while st.nuclei is None:
        pump(0.5)
        if time.time() - t0 > 900:
            raise SystemExit("StarDist did not finish in 15 min")
    pump(1.0)
    print(f"  StarDist: {len(st.nuclei.centroids)} nuclei ({time.time() - t0:.0f} s)")
    a.save_panel("Nuclei segmentation", "nuclei")
    a.click("Run classification")
    scores = st.seeds.scores if st.seeds is not None else None
    theta = float(np.percentile(scores, 85)) if scores is not None and scores.size else 0.0
    a.set_slider("Seed threshold", theta)
    a.save_panel("Object classification", "classification")
    kept = st.nuclei.centroids[st.seeds.kept_indices]
    focus = busiest_point(kept, a.px)
    a.show_only("Signal", "Nuclei", "Nuclei (mask)", "Object seeds")
    a.layer("Object seeds").size = 14
    a.only_open("Object classification")
    a.look(focus, 300)
    a.save_window("nuclei")
    a.close()


GROUPS = {"signal": shots_signal, "nuclei": shots_nuclei}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    words = [w.lower() for w in sys.argv[2:]]
    os.makedirs(OUT, exist_ok=True)
    print("dominion from", dominion.__file__)
    for name, fn in GROUPS.items():
        if words and not any(w in name for w in words):
            continue
        print(name)
        fn(path)


if __name__ == "__main__":
    main()
