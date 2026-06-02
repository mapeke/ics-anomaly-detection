"""Build the 15-minute diploma-defense PowerPoint deck.

Reproducible generator for ``ICS_Anomaly_Detection_Defense.pptx``.

Run:
    python presentation/build_defense_deck.py

The deck reuses the *real* result figures committed under ``results/figures/`` and
the detection numbers in ``results/metrics/summary.csv``; narrative wording follows
the thesis (``Diploma Danial-2.pdf``) and ``presentation/project_overview.tex``.

Design: 16:9, dark-blue title band, single teal accent. The script errors loudly if
a referenced figure is missing rather than embedding a blank placeholder.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"
OUT = REPO / "presentation" / "ICS_Anomaly_Detection_Defense.pptx"

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
NAVY = RGBColor(0x0F, 0x2A, 0x4A)      # title band / headings
TEAL = RGBColor(0x12, 0x9A, 0x8E)      # accent
INK = RGBColor(0x22, 0x2A, 0x33)       # body text
MUTED = RGBColor(0x5B, 0x66, 0x70)     # captions
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)     # code/box background
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ROW_ALT = RGBColor(0xE8, 0xEF, 0xF4)

SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _slide():
    return prs.slides.add_slide(BLANK)


def _box(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    return tb, tf


def _rect(slide, left, top, width, height, fill):
    from pptx.enum.shapes import MSO_SHAPE

    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def _set(run, size, color=INK, bold=False, italic=False, font="Calibri"):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def _heading(slide, title, kicker=None):
    """Standard content-slide header: navy band + title + optional kicker."""
    _rect(slide, 0, 0, SW, Inches(1.15), NAVY)
    _rect(slide, 0, Inches(1.15), SW, Pt(4), TEAL)
    _, tf = _box(slide, Inches(0.55), Inches(0.18), Inches(12.2), Inches(0.95))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    _set(r, 30, WHITE, bold=True)
    if kicker:
        kp = tf.add_paragraph()
        kr = kp.add_run()
        kr.text = kicker
        _set(kr, 13, RGBColor(0xBF, 0xE4, 0xE0), italic=True)


def _footer(slide, idx):
    _, tf = _box(slide, Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.35))
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = f"ICS Anomaly Detection for Cyber-Physical Security  ·  IITU 2026  ·  {idx}/18"
    _set(r, 9, MUTED)


def _bullets(slide, items, left=Inches(0.65), top=Inches(1.5),
             width=Inches(12.0), height=Inches(5.2), size=18, gap=8):
    _, tf = _box(slide, left, top, width, height)
    for i, it in enumerate(items):
        # it: str  OR  (level, text)  OR  (level, text, bold)
        level, text, bold = 0, it, False
        if isinstance(it, tuple):
            level, text = it[0], it[1]
            bold = it[2] if len(it) > 2 else False
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(gap)
        bullet = "•  " if level == 0 else "–  "
        br = p.add_run()
        br.text = bullet
        _set(br, size, TEAL if level == 0 else MUTED, bold=True)
        r = p.add_run()
        r.text = text
        _set(r, size if level == 0 else size - 2, INK, bold=bold)
    return tf


def _caption(slide, text, top):
    _, tf = _box(slide, Inches(0.4), top, Inches(12.5), Inches(0.45))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    _set(r, 13, MUTED, italic=True)


def _image_fit(slide, img_path: Path, max_w, max_h, top, left=None):
    """Add image scaled to fit within (max_w, max_h), centered horizontally."""
    if not img_path.exists():
        raise FileNotFoundError(f"Required figure missing: {img_path}")
    from PIL import Image

    with Image.open(img_path) as im:
        iw, ih = im.size
    ar = iw / ih
    w, h = max_w, int(max_w / ar)
    if h > max_h:
        h, w = max_h, int(max_h * ar)
    if left is None:
        left = int((SW - w) / 2)
    slide.shapes.add_picture(str(img_path), left, top, width=w, height=h)
    return left, top, w, h


def _code(slide, code_text, left, top, width, height, size=11):
    box = _rect(slide, left, top, width, height, LIGHT)
    box.line.color.rgb = RGBColor(0xCF, 0xD8, 0xE0)
    box.line.width = Pt(0.75)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(10)
    tf.margin_right = Pt(8)
    tf.margin_top = Pt(8)
    tf.margin_bottom = Pt(8)
    for i, line in enumerate(code_text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = line if line else " "
        _set(r, size, INK, font="Consolas")
        if line.strip().startswith("#"):
            r.font.color.rgb = TEAL


def _table(slide, rows, left, top, width, height, header=True,
           col_widths=None, size=14, header_size=14):
    nrows, ncols = len(rows), len(rows[0])
    gfx = slide.shapes.add_table(nrows, ncols, left, top, width, height)
    tbl = gfx.table
    if col_widths:
        total = sum(col_widths)
        for j, cw in enumerate(col_widths):
            tbl.columns[j].width = Emu(int(width * cw / total))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Pt(7)
            cell.margin_right = Pt(7)
            cell.margin_top = Pt(3)
            cell.margin_bottom = Pt(3)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            if header and i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
                _set(r, header_size, WHITE, bold=True)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if (i % 2 == 1) else ROW_ALT
                _set(r, size, INK, bold=(j == 0))
    return tbl


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------- #
# Slide 1 — Title
# --------------------------------------------------------------------------- #
def slide_title():
    s = _slide()
    _rect(s, 0, 0, SW, SH, NAVY)
    _rect(s, 0, Inches(3.05), SW, Pt(4), TEAL)

    _, tf = _box(s, Inches(0.8), Inches(0.55), Inches(11.7), Inches(0.6))
    r = tf.paragraphs[0].add_run()
    r.text = "International Information Technology University · Faculty of Computer Technology and Cybersecurity"
    _set(r, 14, RGBColor(0xBF, 0xE4, 0xE0))

    _, tf = _box(s, Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.6))
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "ICS Anomaly Detection\nfor Cyber-Physical Security"
    _set(r, 46, WHITE, bold=True)

    _, tf = _box(s, Inches(0.8), Inches(3.25), Inches(11.7), Inches(0.6))
    r = tf.paragraphs[0].add_run()
    r.text = "Diploma Project  ·  Educational Program 6B06303 — Network Security"
    _set(r, 18, TEAL, bold=True)

    _, tf = _box(s, Inches(0.8), Inches(4.15), Inches(11.7), Inches(2.0))
    lines = [
        ("Team lead:  Zainiddin Danial", True),
        ("Imankozhayeva Laura", False),
        ("Bekisheva Galiya", False),
        ("Research advisor:  Nurlybayev T. A.", False),
    ]
    for i, (txt, bold) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(4)
        r = p.add_run()
        r.text = txt
        _set(r, 18, WHITE, bold=bold)

    _, tf = _box(s, Inches(0.8), Inches(6.7), Inches(11.7), Inches(0.5))
    r = tf.paragraphs[0].add_run()
    r.text = "Almaty 2026"
    _set(r, 16, RGBColor(0xBF, 0xE4, 0xE0))
    _notes(s, "0:00–0:40 — Greet the committee, read the topic, introduce the three of us "
               "and our advisor. Danial leads the talk.")


# --------------------------------------------------------------------------- #
# Slide 2 — Team & roles
# --------------------------------------------------------------------------- #
def slide_team(idx):
    s = _slide()
    _heading(s, "Team composition & roles",
             "Three students, three contributions of the thesis")
    rows = [
        ["Member", "Role", "Owns"],
        ["Zainiddin Danial\n(team lead)", "ML Engineering",
         "Unified AnomalyDetector interface; the six detectors; experiment "
         "infrastructure & reproducibility (YAML configs, config hash)"],
        ["Imankozhayeva Laura", "Data & Evaluation",
         "Data loaders & preprocessing (scaler-leak guard); three time-aware "
         "metrics; cross-testbed transfer + within-HAI LOPO study"],
        ["Bekisheva Galiya", "Attribution & Delivery",
         "Per-sensor attribution + SHAP; FastAPI demo web app; thesis writing "
         "and figures"],
        ["Nurlybayev T. A.", "Research advisor",
         "Scope, methodology review, defense readiness"],
    ]
    _table(s, rows, Inches(0.55), Inches(1.55), Inches(12.2), Inches(4.6),
           col_widths=[2.4, 2.2, 7.0], size=14)
    _footer(s, idx)
    _notes(s, "0:40–1:30 — Who did what. Each role maps to one of the three "
               "contributions: detection benchmark, cross-dataset transfer, attribution.")


# --------------------------------------------------------------------------- #
# Slide 3 — Motivation
# --------------------------------------------------------------------------- #
def slide_motivation(idx):
    s = _slide()
    _heading(s, "Why this matters: attacks on physical plants",
             "Industrial Control Systems run water, power, gas, chemicals")
    # data-flow strip
    chain = ["Sensors\n(pressure, flow,\nvalve, motor)", "ICS / PLC\ncontrol logic",
             "Actuators\n(pumps, valves,\nbreakers)"]
    bw, bh, gap = Inches(2.4), Inches(1.05), Inches(0.55)
    x = Inches(0.85)
    y = Inches(1.55)
    for i, c in enumerate(chain):
        shp = _rect(s, x, y, bw, bh, TEAL if i == 1 else NAVY)
        tf = shp.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = c
        _set(r, 13, WHITE, bold=True)
        if i < 2:
            _, atf = _box(s, x + bw, y + Inches(0.3), gap, Inches(0.5))
            ap = atf.paragraphs[0]
            ap.alignment = PP_ALIGN.CENTER
            ar = ap.add_run()
            ar.text = "→"
            _set(ar, 26, MUTED, bold=True)
        x = x + bw + gap
    # monitoring callout
    mon = _rect(s, Inches(0.85), Inches(2.75), bw * 3 + gap * 2, Inches(0.55), LIGHT)
    mtf = mon.text_frame
    mp = mtf.paragraphs[0]
    mp.alignment = PP_ALIGN.CENTER
    mr = mp.add_run()
    mr.text = "Anomaly detection lives in the monitoring layer:  learn what \"normal\" looks like → alarm on \"not normal\""
    _set(mr, 13, NAVY, bold=True)

    _bullets(s, [
        "Stuxnet (2010) — malware physically destroyed centrifuges while screens showed normal readings",
        "Ukraine power grid (2015) — attackers tripped breakers, ~225,000 customers blacked out in winter",
        "Maroochy Shire (2000) — stolen radio opened sewage valves, ~1 million litres spilled",
        (0, "In every case the attack was visible in the sensor data — if someone watched the right pattern.", True),
    ], top=Inches(3.55), size=17)
    _footer(s, idx)
    _notes(s, "1:30–2:40 — Set the stakes: ICS attacks have physical consequences. "
               "The premise of the field: teach a model 'normal', flag 'abnormal'.")


# --------------------------------------------------------------------------- #
# Slide 4 — Task statement
# --------------------------------------------------------------------------- #
def slide_task(idx):
    s = _slide()
    _heading(s, "Task statement", "What the diploma sets out to prove and build")
    box = _rect(s, Inches(0.55), Inches(1.45), Inches(12.2), Inches(1.35), LIGHT)
    box.line.color.rgb = TEAL
    box.line.width = Pt(1.5)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(14)
    tf.margin_top = Pt(8)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "Thesis: "
    _set(r, 15, TEAL, bold=True)
    r2 = p.add_run()
    r2.text = ("ICS detectors report high F1 on their training dataset, but (a) their "
               "rankings change under time-aware metrics and (b) they fail to generalize "
               "across testbeds. We quantify both effects on HAI and Morris, and add "
               "per-sensor attribution to evaluate detection AND localization.")
    _set(r2, 15, INK)

    _bullets(s, [
        "1. Survey ICS anomaly-detection methods, datasets, and time-aware evaluation metrics",
        "2. Implement six detectors under one AnomalyDetector interface (fit / score / attribute)",
        "3. Implement three metrics with unit tests; reproduce published baselines within tolerance",
        "4. Cross-testbed transfer HAI ↔ Morris (source- vs target-calibrated) + within-HAI leave-one-process-out",
        "5. Per-sensor attribution vs HAI process-level attack labels, with multi-seed sensitivity",
        "6. Findings, practical recommendations, and a reproducible apparatus",
    ], top=Inches(3.0), size=16, gap=7)
    _footer(s, idx)
    _notes(s, "2:40–3:50 — The argument in one box, then the six concrete objectives "
               "straight from the assignment. This frames everything that follows.")


# --------------------------------------------------------------------------- #
# Slide 5 — Datasets
# --------------------------------------------------------------------------- #
def slide_datasets(idx):
    s = _slide()
    _heading(s, "The two testbeds", "Deliberately dissimilar — that is the point")
    rows = [
        ["", "HAI 21.03", "Morris (gas pipeline)"],
        ["Domain", "Power-plant (boiler, turbine, water)", "Gas-pipeline Modbus RTU"],
        ["Features", "79 sensor channels", "16 features (leak cols removed)"],
        ["Processes", "4 interconnected (P1–P4)", "Single pipeline"],
        ["Rate", "1 Hz, timestamped", "Sub-second, no timestamps"],
        ["Format", "Folder of CSVs", "ARFF file"],
        ["Shared sensors", "— none in common —", "— none in common —"],
    ]
    _table(s, rows, Inches(0.55), Inches(1.55), Inches(7.1), Inches(4.4),
           col_widths=[1.7, 3.0, 3.0], size=12.5, header_size=13)
    _image_fit(s, FIG / "01_exploration" / "hai_overview.png",
               max_w=Inches(5.2), max_h=Inches(4.0), top=Inches(1.75),
               left=Inches(7.9))
    _caption(s, "HAI process overview — 79 channels across four physical loops",
             top=Inches(5.9))
    _footer(s, idx)
    _notes(s, "3:50–4:40 — Two factories that don't speak the same language: "
               "79 power-plant sensors vs 16 gas-pipeline Modbus features, zero shared sensors. "
               "This is what makes cross-dataset transfer hard.")


# --------------------------------------------------------------------------- #
# Slide 6 — Six models
# --------------------------------------------------------------------------- #
def slide_models(idx):
    s = _slide()
    _heading(s, "Six detectors, one interface",
             "Two classical · two autoencoders · two transformer-era SOTA")
    rows = [
        ["Model", "Family", "Year", "Core idea"],
        ["Isolation Forest", "Classical", "2008", "Random splits isolate outliers faster"],
        ["One-Class SVM", "Classical", "2001", "Boundary around the normal region"],
        ["Dense Autoencoder", "Autoencoder", "—", "High reconstruction error = anomaly"],
        ["LSTM Autoencoder", "Autoencoder", "2016", "Sequence reconstruction over time"],
        ["USAD", "SOTA", "2020", "Adversarially-trained autoencoder pair"],
        ["TranAD", "SOTA", "2022", "Transformer with attention + adversarial loss"],
    ]
    _table(s, rows, Inches(0.7), Inches(1.6), Inches(11.9), Inches(4.4),
           col_widths=[2.4, 1.6, 1.0, 5.5], size=14)
    _footer(s, idx)
    _notes(s, "4:40–5:30 — The lineup spans 20+ years. The whole point of the shared "
               "interface is that swapping any of these is a one-line config change — "
               "apples-to-apples by construction.")


# --------------------------------------------------------------------------- #
# Slide 7 — Metrics
# --------------------------------------------------------------------------- #
def slide_metrics(idx):
    s = _slide()
    _heading(s, "Three time-aware metrics",
             "How you score a detector changes who wins")
    cards = [
        ("Point-wise F1", "Standard per-timestep precision/recall/F1.",
         "Honest but unforgiving on short events.", NAVY),
        ("Point-adjust F1", "If the model flags ANY point in an attack, the whole "
         "attack counts as detected.",
         "Inflated — a random flagger beats real detectors (Kim et al., AAAI 2022).", RGBColor(0xB0, 0x55, 0x2B)),
        ("eTaPR", "Event-aware: rewards how much of each attack you catch and how fast.",
         "The honest metric we trust for ranking.", TEAL),
    ]
    cw = Inches(3.9)
    x = Inches(0.5)
    for title, body, note, col in cards:
        card = _rect(s, x, Inches(1.6), cw, Inches(4.2), WHITE)
        card.line.color.rgb = col
        card.line.width = Pt(2)
        _rect(s, x, Inches(1.6), cw, Inches(0.7), col)
        _, htf = _box(s, x, Inches(1.62), cw, Inches(0.66))
        htf.vertical_anchor = MSO_ANCHOR.MIDDLE
        hp = htf.paragraphs[0]
        hp.alignment = PP_ALIGN.CENTER
        hr = hp.add_run()
        hr.text = title
        _set(hr, 18, WHITE, bold=True)
        _, btf = _box(s, x + Inches(0.2), Inches(2.5), cw - Inches(0.4), Inches(3.1))
        bp = btf.paragraphs[0]
        br = bp.add_run()
        br.text = body
        _set(br, 14, INK)
        np_ = btf.add_paragraph()
        np_.space_before = Pt(12)
        nr = np_.add_run()
        nr.text = note
        _set(nr, 13, col, italic=True, bold=True)
        x = x + cw + Inches(0.35)
    _footer(s, idx)
    _notes(s, "5:30–6:30 — The metric is not neutral. Point-adjust over-credits a single "
               "lucky hit; eTaPR grades on coverage and speed. We report all three because "
               "the ranking changes between them — and that change is a finding.")


# --------------------------------------------------------------------------- #
# Slide 8 — Architecture
# --------------------------------------------------------------------------- #
def slide_architecture(idx):
    s = _slide()
    _heading(s, "System architecture", "Config-driven, reproducible, no hidden state")
    # data-flow pipeline
    steps = ["Raw data\n(HAI / Morris)", "data_loader\n→ unified schema",
             "preprocessing\nscale + window", "Model\nfit · score · attribute",
             "evaluation\n3 metrics", "summary.parquet\n+ figures"]
    bw, bh = Inches(1.92), Inches(1.0)
    x = Inches(0.35)
    y = Inches(1.7)
    for i, st in enumerate(steps):
        shp = _rect(s, x, y, bw, bh, NAVY if i % 2 == 0 else TEAL)
        tf = shp.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = st
        _set(r, 11.5, WHITE, bold=True)
        if i < len(steps) - 1:
            _, atf = _box(s, x + bw - Inches(0.02), y + Inches(0.3), Inches(0.3), Inches(0.4))
            ar = atf.paragraphs[0].add_run()
            ar.text = "→"
            _set(ar, 20, MUTED, bold=True)
        x = x + bw + Inches(0.18)

    _bullets(s, [
        "One AnomalyDetector base class — fit / score / attribute / save / load — every model obeys it",
        "Every experiment is a YAML config; the runner is model-agnostic",
        "Each run records a 12-char config hash + seed + timestamp → one row in summary.parquet",
        "Notebooks only plot. If a number is in the thesis, a tested function in src/ produced it",
    ], top=Inches(3.2), size=17)
    _footer(s, idx)
    _notes(s, "6:30–7:30 — Trace one number end to end. The discipline — config hash, "
               "seeds, tests, notebooks-only-plot — is what lets a reviewer trust the numbers.")


# --------------------------------------------------------------------------- #
# Slide 9 — Transfer method
# --------------------------------------------------------------------------- #
def slide_transfer_method(idx):
    s = _slide()
    _heading(s, "Bridging two testbeds: the type vector",
             "If they share no sensors, compare sensor TYPES")
    # HAI box -> 6-dim -> Morris box
    left = _rect(s, Inches(0.7), Inches(2.0), Inches(3.0), Inches(2.4), NAVY)
    ltf = left.text_frame; ltf.word_wrap = True
    lp = ltf.paragraphs[0]; lp.alignment = PP_ALIGN.CENTER
    lr = lp.add_run(); lr.text = "HAI\n79 features"; _set(lr, 18, WHITE, bold=True)

    mid = _rect(s, Inches(5.1), Inches(1.85), Inches(3.1), Inches(2.7), TEAL)
    mtf = mid.text_frame; mtf.word_wrap = True
    mp = mtf.paragraphs[0]; mp.alignment = PP_ALIGN.CENTER
    mr = mp.add_run()
    mr.text = "Shared 6-dim type space"
    _set(mr, 16, WHITE, bold=True)
    for t in ["pressure", "pump_state", "setpoint",
              "valve_position", "control_signal", "system_state"]:
        tp = mtf.add_paragraph(); tp.alignment = PP_ALIGN.CENTER
        tr = tp.add_run(); tr.text = t; _set(tr, 12, WHITE)

    right = _rect(s, Inches(9.6), Inches(2.0), Inches(3.0), Inches(2.4), NAVY)
    rtf = right.text_frame; rtf.word_wrap = True
    rp = rtf.paragraphs[0]; rp.alignment = PP_ALIGN.CENTER
    rr = rp.add_run(); rr.text = "Morris\n16 features"; _set(rr, 18, WHITE, bold=True)

    for ax, ay in [(Inches(3.75), Inches(3.0)), (Inches(8.25), Inches(3.0))]:
        _, atf = _box(s, ax, ay, Inches(1.3), Inches(0.6))
        ar = atf.paragraphs[0]; ar.alignment = PP_ALIGN.CENTER
        a = ar.add_run(); a.text = "→"; _set(a, 30, TEAL, bold=True)

    _bullets(s, [
        "Each sensor hand-tagged by TYPE in data/feature_types.yaml (pressure, valve, control signal…)",
        "Both datasets projected onto the shared 6-dim type vector → models become transferable",
        "Transfer the scoring function, not the raw weights — methodologically the cleanest option",
    ], top=Inches(4.9), size=16)
    _footer(s, idx)
    _notes(s, "7:30–8:20 — The core methodological move. We don't match 'sensor 47' to "
               "'sensor 11'; we match 'pressure behaviour' to 'pressure behaviour'.")


# --------------------------------------------------------------------------- #
# Slide 10 — Software & reproducibility
# --------------------------------------------------------------------------- #
def slide_software(idx):
    s = _slide()
    _heading(s, "Software & reproducibility", "Tooling and the build-everything-from-config rule")
    _bullets(s, [
        (0, "Languages & ML", True),
        (1, "Python 3.11 · PyTorch 2.3 (deep models) · scikit-learn 1.4 (classical) · NumPy / pandas"),
        (0, "Explainability & app", True),
        (1, "SHAP (classical attribution) · FastAPI + vanilla JS demo web app · joblib artifacts"),
        (0, "Quality & docs", True),
        (1, "pytest 8.0 (13 test modules) · ruff · pre-commit · LaTeX (thesis + this deck's figures)"),
        (0, "Reproducibility contract", True),
        (1, "python -m experiments.run <config.yaml>  →  one row in results/metrics/summary.parquet"),
        (1, "~48 YAML configs · 12-char config hash · seeds 7 / 42 / 123 · Parquet for nested metrics"),
    ], top=Inches(1.55), size=17, gap=6)
    _footer(s, idx)
    _notes(s, "8:20–9:00 — The stack, and the one rule that makes it a research artifact: "
               "every number reproducible from a committed config. Quick — keep moving to results.")


# --------------------------------------------------------------------------- #
# Slide 11 — Implementation
# --------------------------------------------------------------------------- #
def slide_implementation(idx):
    s = _slide()
    _heading(s, "Implementation — key points",
             "The interface contract and the anti-leakage guard")
    _, t1 = _box(s, Inches(0.55), Inches(1.35), Inches(6.0), Inches(0.4))
    r = t1.paragraphs[0].add_run()
    r.text = "src/models/base.py — every model obeys this"
    _set(r, 14, NAVY, bold=True)
    _code(s, (
        "class AnomalyDetector(ABC):\n"
        "    @abstractmethod\n"
        "    def fit(self, X_train, X_val=None):\n"
        "        # Fit on normal-only data.\n"
        "        # Must never see attack rows.\n"
        "        ...\n"
        "    @abstractmethod\n"
        "    def score(self, X):\n"
        "        # per-sample anomaly score\n"
        "        ...\n"
        "    def attribute(self, X):\n"
        "        # per-feature contribution\n"
        "        ..."
    ), Inches(0.55), Inches(1.75), Inches(6.0), Inches(3.6), size=12)

    _, t2 = _box(s, Inches(6.85), Inches(1.35), Inches(6.0), Inches(0.4))
    r = t2.paragraphs[0].add_run()
    r.text = "src/preprocessing.py — scaler-leak guard"
    _set(r, 14, NAVY, bold=True)
    _code(s, (
        "def scale_bundle(bundle):\n"
        "    # #1 cause of inflated ICS results:\n"
        "    # fitting the scaler on attack rows.\n"
        "    # Crash rather than trust callers.\n"
        "    bundle.assert_no_attack_in_train_val()\n"
        "\n"
        "    scaler = MinMaxScaler(clip=True)\n"
        "    scaler.fit(X_train)   # normal only\n"
        "    return ScaledArrays(...)"
    ), Inches(6.85), Inches(1.75), Inches(6.0), Inches(2.3), size=12)

    box = _rect(s, Inches(6.85), Inches(4.25), Inches(6.0), Inches(1.55), LIGHT)
    box.line.color.rgb = TEAL; box.line.width = Pt(1.25)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Pt(10); tf.margin_top = Pt(8)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = "Demo web app  "
    _set(r, 14, TEAL, bold=True)
    r2 = p.add_run()
    r2.text = "(FastAPI + vanilla JS)"
    _set(r2, 13, MUTED, italic=True)
    p2 = tf.add_paragraph()
    r3 = p2.add_run()
    r3.text = ("Upload a CSV → pick a trained artifact → see per-row anomaly scores. "
               "The CLI and the app are thin wrappers around one function: "
               "score_dataframe(artifact, df).")
    _set(r3, 13, INK)
    _footer(s, idx)
    _notes(s, "9:00–10:00 — Two code excerpts that ARE the argument: the shared interface "
               "(left) and the assertion that crashes on data leakage (right). Plus the demo "
               "app that turns a trained model into something you can click.")


# --------------------------------------------------------------------------- #
# Slide 12 — Detection grid
# --------------------------------------------------------------------------- #
def slide_detection(idx):
    s = _slide()
    _heading(s, "Results — same-dataset detection",
             "Point-wise F1 / ROC-AUC, scaler fit on normal only")
    rows = [
        ["Model", "HAI F1", "HAI AUC", "Morris F1", "Morris AUC"],
        ["Dense AE", "0.524", "0.859", "0.737", "0.751"],
        ["Isolation Forest", "0.208", "0.770", "0.734", "0.658"],
        ["One-Class SVM", "0.307", "0.757", "0.653", "0.629"],
        ["LSTM AE", "0.148", "0.670", "0.9997", "—"],
    ]
    _table(s, rows, Inches(0.55), Inches(1.6), Inches(6.3), Inches(3.0),
           col_widths=[2.3, 1.0, 1.0, 1.1, 1.1], size=13)
    _, tf = _box(s, Inches(0.55), Inches(4.9), Inches(6.3), Inches(1.9))
    _bullets(s, [
        "Dense AE leads on HAI; absolute F1 is modest (0.52) even at home",
        "High Morris F1 partly reflects its ~91% windowed attack rate — a warning sign",
    ], left=Inches(0.55), top=Inches(4.9), width=Inches(6.3), size=14)
    _image_fit(s, FIG / "04_sota" / "sota_six_model_comparison.png",
               max_w=Inches(6.0), max_h=Inches(4.8), top=Inches(1.7),
               left=Inches(7.1))
    _caption(s, "Six-model comparison across metrics", top=Inches(6.55))
    _footer(s, idx)
    _notes(s, "10:00–10:55 — Baseline detection. Note nobody hits the 0.95 the literature "
               "advertises once leakage is closed. Morris's high F1 is a base-rate artifact — "
               "foreshadows the transfer finding.")


# --------------------------------------------------------------------------- #
# Slide 13 — Metric sensitivity
# --------------------------------------------------------------------------- #
def slide_metric_sensitivity(idx):
    s = _slide()
    _heading(s, "Results — metric sensitivity",
             "The same models, re-ranked by the metric")
    _image_fit(s, FIG / "06_metric_sensitivity" / "metric_sensitivity.png",
               max_w=Inches(8.2), max_h=Inches(4.9), top=Inches(1.5),
               left=Inches(0.5))
    _bullets(s, [
        (0, "Rankings shuffle", True),
        (1, "between point-wise, PA-F1, eTaPR"),
        (0, "PA-F1 inflates", True),
        (1, "classical detectors most"),
        (0, "Takeaway", True),
        (1, "a single-metric leaderboard is unreliable"),
    ], left=Inches(8.9), top=Inches(1.7), width=Inches(4.0), size=15, gap=5)
    _caption(s, "Metric choice reorders the leaderboard", top=Inches(6.55))
    _footer(s, idx)
    _notes(s, "10:55–11:40 — Contribution 2 in one chart: who 'wins' depends on the metric. "
               "Point-adjust flatters the weak classical models the most.")


# --------------------------------------------------------------------------- #
# Slide 14 — Cross-dataset transfer (headline)
# --------------------------------------------------------------------------- #
def slide_transfer_results(idx):
    s = _slide()
    _heading(s, "Results — cross-testbed transfer  (headline)",
             "Calibration, not the model, decides whether transfer means anything")
    _image_fit(s, FIG / "05_cross_dataset" / "transfer_calibration_comparison.png",
               max_w=Inches(6.1), max_h=Inches(4.5), top=Inches(1.6),
               left=Inches(0.45))
    _image_fit(s, FIG / "05_cross_dataset" / "lopo_heatmap.png",
               max_w=Inches(6.0), max_h=Inches(4.5), top=Inches(1.6),
               left=Inches(6.85))
    _bullets(s, [
        (0, "Source-calibrated transfer collapses every detector to the target's class-prior baseline "
            "(F1≈0.95 on Morris, ≈0.05 on HAI) — the number reflects base rates, not intelligence.", True),
        "Target-calibrating the threshold on normal target data restores meaning: SOTA reaches "
        "0.38–0.44 eTaPR (HAI→Morris) while classical baselines stay below 0.13.",
        "Within-HAI LOPO (right): hiding a process's sensors causes targeted blindness, not uniform decay.",
    ], top=Inches(5.45), size=13.5, gap=4)
    _footer(s, idx)
    _notes(s, "11:40–12:55 — The core novelty. Left: naive transfer is a coin flip dressed up as "
               "0.95. Recalibrate on target normal and a real modern-vs-classical gap appears. "
               "Right: LOPO shows blindness is localized to the removed subsystem.")


# --------------------------------------------------------------------------- #
# Slide 15 — Attribution
# --------------------------------------------------------------------------- #
def slide_attribution(idx):
    s = _slide()
    _heading(s, "Results — attribution & localization",
             "\"We caught it\" ≠ \"here's which sensor\"")
    _image_fit(s, FIG / "07_attribution" / "attribution_p_at_5.png",
               max_w=Inches(6.0), max_h=Inches(4.4), top=Inches(1.6),
               left=Inches(0.45))
    _image_fit(s, FIG / "07_attribution" / "shap_if_ocsvm_pak.png",
               max_w=Inches(6.0), max_h=Inches(4.4), top=Inches(1.6),
               left=Inches(6.9))
    _bullets(s, [
        (0, "Detection-vs-localization inversion: the Dense AE — weakest by F1 — is the only "
            "windowed/AE model above random on all three attacked HAI processes.", True),
        "Attention rollout (TranAD) is numerically identical to reconstruction error — a null result.",
        "SHAP lifts the two classical baselines above random too (right).",
    ], top=Inches(5.4), size=13.5, gap=4)
    _footer(s, idx)
    _notes(s, "12:55–13:50 — Contribution 3. The best detector is often the worst localizer. "
               "The fancy attention explanation adds nothing over plain reconstruction error.")


# --------------------------------------------------------------------------- #
# Slide 16 — Findings & recommendations
# --------------------------------------------------------------------------- #
def slide_findings(idx):
    s = _slide()
    _heading(s, "Key findings & recommendations",
             "Three findings → three things a security team should do")
    # left: findings
    _, tf = _box(s, Inches(0.55), Inches(1.5), Inches(6.1), Inches(5.2))
    head = tf.paragraphs[0].add_run(); head.text = "Findings"
    _set(head, 18, TEAL, bold=True)
    for t in [
        "1. Naive cross-factory transfer collapses to coin-flipping (class-prior baseline).",
        "2. Hiding one subsystem's sensors causes targeted blindness, not global decay.",
        "3. The worst detector (Dense AE) is the best localizer; attention explanations are null.",
    ]:
        p = tf.add_paragraph(); p.space_before = Pt(10)
        r = p.add_run(); r.text = t; _set(r, 15, INK)
    # right: recommendations
    box = _rect(s, Inches(6.9), Inches(1.5), Inches(5.9), Inches(5.2), LIGHT)
    box.line.color.rgb = TEAL; box.line.width = Pt(1.5)
    tf2 = box.text_frame; tf2.word_wrap = True
    tf2.margin_left = Pt(14); tf2.margin_top = Pt(12); tf2.margin_right = Pt(12)
    h2 = tf2.paragraphs[0].add_run(); h2.text = "Recommendations"
    _set(h2, 18, NAVY, bold=True)
    for t in [
        "Calibrate the threshold on target-domain normal data before deploying a transferred detector.",
        "Report multiple time-aware metrics (incl. eTaPR); distrust single-metric rankings.",
        "Audit attribution alongside detection — consider a SOTA detector + Dense-AE/SHAP localizer hybrid.",
    ]:
        p = tf2.add_paragraph(); p.space_before = Pt(12)
        r = p.add_run(); r.text = "✓  " + t; _set(r, 15, INK)
    _footer(s, idx)
    _notes(s, "13:50–14:30 — Tie the three findings to three concrete, actionable rules. "
               "This is the 'so what' for a practitioner or reviewer.")


# --------------------------------------------------------------------------- #
# Slide 17 — Conclusion
# --------------------------------------------------------------------------- #
def slide_conclusion(idx):
    s = _slide()
    _heading(s, "Conclusion", "A measurement project, shipped reproducibly")
    _bullets(s, [
        "Benchmarked six detectors on HAI and Morris under three time-aware metrics",
        "Separated representation transfer from threshold transfer — exposing how published F1 misleads",
        "Established a detection-vs-localization tradeoff via per-sensor attribution",
        (0, "Contributions:", True),
        (1, "A reproducible benchmarking protocol — every number rebuilds from a committed YAML"),
        (1, "An open-source implementation under one detector interface (7th model = one class)"),
        (1, "A recommendation to the field: report the full grid — metrics × calibration × attribution — "
            "not a single headline F1"),
    ], top=Inches(1.6), size=17, gap=8)
    _footer(s, idx)
    _notes(s, "14:30–14:55 — Restate what we did and the single message: report the grid, "
               "not the headline. Mention the apparatus outlives the thesis.")


# --------------------------------------------------------------------------- #
# Slide 18 — Thank you
# --------------------------------------------------------------------------- #
def slide_thanks():
    s = _slide()
    _rect(s, 0, 0, SW, SH, NAVY)
    _rect(s, 0, Inches(4.0), SW, Pt(4), TEAL)
    _, tf = _box(s, Inches(0.8), Inches(2.5), Inches(11.7), Inches(1.4))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "Thank you for your attention"
    _set(r, 44, WHITE, bold=True)
    _, tf = _box(s, Inches(0.8), Inches(4.3), Inches(11.7), Inches(1.6))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "ICS Anomaly Detection for Cyber-Physical Security"
    _set(r, 20, TEAL, bold=True)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(14)
    r2 = p2.add_run()
    r2.text = "Zainiddin Danial · Imankozhayeva Laura · Bekisheva Galiya"
    _set(r2, 18, WHITE)
    p3 = tf.add_paragraph(); p3.alignment = PP_ALIGN.CENTER
    r3 = p3.add_run()
    r3.text = "Advisor: Nurlybayev T. A.  ·  IITU, Almaty 2026"
    _set(r3, 15, RGBColor(0xBF, 0xE4, 0xE0))
    _notes(s, "14:55–15:00 — Thank the committee. Invite questions.")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def main():
    slide_title()
    slide_team(2)
    slide_motivation(3)
    slide_task(4)
    slide_datasets(5)
    slide_models(6)
    slide_metrics(7)
    slide_architecture(8)
    slide_transfer_method(9)
    slide_software(10)
    slide_implementation(11)
    slide_detection(12)
    slide_metric_sensitivity(13)
    slide_transfer_results(14)
    slide_attribution(15)
    slide_findings(16)
    slide_conclusion(17)
    slide_thanks()

    assert len(prs.slides._sldIdLst) == 18, "expected 18 slides"
    prs.save(str(OUT))
    print(f"Wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
