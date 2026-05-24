"""Build 5-slide contest deck.

Genera C:/Users/Antoine/Desktop/Progetti/GSW/presentazione_contest.pptx
con titolo + 1 immagine grande + bullet minimi + speaker notes complete
(lette dal SLIDE_NOTES.md a mano e incollate qui — niente parsing).

Palette: Ocean Gradient (marittimo).
Font: Georgia (titolo) + Calibri (body).
Layout: 16:9 (13.333" x 7.5") via LAYOUT_WIDE-equivalent.
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT_PATH = ROOT / "presentazione_contest.pptx"

# ============================================================================
# Palette — Ocean Gradient
# ============================================================================
COLOR_DEEP_BLUE = RGBColor(0x06, 0x5A, 0x82)   # primary
COLOR_TEAL      = RGBColor(0x1C, 0x72, 0x93)   # secondary
COLOR_MIDNIGHT  = RGBColor(0x21, 0x29, 0x5C)   # accent (titoli)
COLOR_BG_LIGHT  = RGBColor(0xFF, 0xFF, 0xFF)   # body bg
COLOR_BG_DARK   = RGBColor(0x0F, 0x1B, 0x3D)   # title/conclusion bg
COLOR_TEXT_DARK = RGBColor(0x1A, 0x1A, 0x1A)
COLOR_TEXT_MUTE = RGBColor(0x55, 0x66, 0x77)
COLOR_ACCENT    = RGBColor(0xFF, 0xA5, 0x00)   # warning/honesty highlights
COLOR_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

FONT_TITLE = "Georgia"
FONT_BODY  = "Calibri"


# ============================================================================
# Slide content (5 slide, speaker notes complete in italiano)
# ============================================================================
SLIDES = [
    # ---- Slide 1: Problema ----
    {
        "kind": "content_dark",
        "title": "Dark Fleet",
        "subtitle": "Quando il segnale è l'assenza di segnale",
        "image": str(MODELS / "06_geographic_map.png"),
        "bullets": [
            "AIS spento intenzionalmente → pesca illegale, trasbordi, sanzioni",
            "Definiamo evento dark = gap AIS > 12h in area ad alta copertura sat.",
            "Domanda: prevediamo il blackout dalla cinematica delle 2h precedenti?",
        ],
        "notes": (
            "Le navi commerciali trasmettono via AIS la propria posizione ogni 10 "
            "secondi. Tutte tranne una categoria: la dark fleet — navi che spengono "
            "intenzionalmente il transponder per nascondere pesca illegale, "
            "trasbordi clandestini, violazioni di sanzioni.\n\n"
            "Non possiamo etichettare il crimine direttamente — niente ground truth "
            "giudiziaria. Quello che possiamo modellare è il precursore tattico: lo "
            "spegnimento del transponder. Definiamo evento dark = gap AIS maggiore "
            "di 12 ore in area ad alta copertura satellitare, lo Stretto di Sicilia. "
            "La letteratura — in particolare Global Fishing Watch — supporta che a "
            "12 ore o più il gap è overwhelmingly intenzionale, non un guasto.\n\n"
            "La domanda diventa: possiamo prevedere il blackout dalla cinematica "
            "della nave nelle ore precedenti?"
        ),
    },
    # ---- Slide 2: Approccio ----
    {
        "kind": "content_light",
        "title": "Hybrid: Bayesian Mixture + Gradient Boosting",
        "subtitle": "Full Bayesian rigour, pipeline causale",
        "image": str(MODELS / "00b_empirical_bayes_priors.png"),
        "bullets": [
            "BMM in PyMC, ADVI, finestra causale 36 ping → P(regime sospetto)",
            "Empirical Bayes 2-stage (Efron-Morris): pilot fit → hyperprior",
            "LightGBM + Optuna su feature cinematica + posterior Bayesiana",
            "Validazione: PPC, Prior Sensitivity, group-disjoint CV",
        ],
        "notes": (
            "Pipeline in tre stadi, isolati per prevenire leakage temporale.\n\n"
            "Uno: feature engineering causale — accelerazione longitudinale, rate "
            "of turn, dt fra ping.\n\n"
            "Due: Bayesian Mixture Model in PyMC con ADVI su finestra rolling "
            "causale di 36 ping. Estrae la probabilità del regime sospetto in "
            "ogni istante. Two-stage Empirical Bayes alla Efron-Morris: gli "
            "hyperprior sigma sono stimati da fit pilota su 3 navi rappresentative, "
            "poi posterior individuale per ognuna delle 120 navi. Markov filter "
            "post-hoc, complessità lineare, per imporre dipendenza temporale "
            "senza overhead MCMC.\n\n"
            "Tre: LightGBM ottimizzato con Optuna, 10 trial, 3 fold, che riceve "
            "le feature cinematiche più la probabilità Bayesiana smoothed.\n\n"
            "Validazione triplice: Posterior Predictive Check alla Gelman BDA3 "
            "capitolo 6, Prior Sensitivity Analysis, e group-disjoint CV che "
            "impedisce al modello di imparare il fingerprint della singola nave."
        ),
    },
    # ---- Slide 3: Risultato ----
    {
        "kind": "content_light",
        "title": "Risultato",
        "subtitle": "PR-AUC in CV = 0.19 = 44× baseline",
        "image": str(MODELS / "05_pr_curve_with_thresholds.png"),
        "bullets": [
            "CV group-disjoint: PR-AUC 0.19 (baseline 0.43%)",
            "Test holdout (24 navi): PR-AUC 0.005 ≈ baseline",
            "Soglia F2-ottimale: recall 19% • Conformal FPR ≤ 10%: recall 12%",
            "Gap CV→test: alta varianza su rare events con piccolo holdout",
        ],
        "notes": (
            "Risultato chiave in cross-validation group-disjoint: PR-AUC pari a "
            "0.190, contro una baseline di prevalenza dello 0.43%. Il lift è 44 "
            "volte sopra il random. Il modello impara un segnale forte.\n\n"
            "Sul test holdout — 24 navi mai viste in training — PR-AUC scende a "
            "0.005, vicino al baseline. ROC-AUC 0.53. Questa è la realtà che "
            "vogliamo presentare onestamente: il gap CV verso test è il sintomo "
            "classico di alta varianza su rare events con holdout piccolo. 24 navi "
            "moltiplicato per una prevalenza dello 0.43 percento fa circa 190 "
            "positivi assoluti, troppo pochi per stabilizzare la stima.\n\n"
            "A livello operativo, scegliendo la soglia F2-ottimale (β=2, peso "
            "recall): recall 19 percento, precision marginale. A soglia conformal "
            "con FPR garantito sotto il 10 percento, recall 12 percento."
        ),
    },
    # ---- Slide 4: Credibilità ----
    {
        "kind": "content_light",
        "title": "Rigour reveals what naïve evaluation hides",
        "subtitle": "Validazione critica, anche quando dice cose scomode",
        "image": str(MODELS / "00d_posterior_predictive_check.png"),
        "bullets": [
            "PPC: modello adeguato su 3/4 statistiche, discrepanza sul tail",
            "Prior Sensitivity: min ρ = −1.0 → modello NON robusto al prior",
            "Bootstrap CI clustered per MMSI (1000 iter): include zero",
            "Tutto questo è scoperta, non fallimento.",
        ],
        "notes": (
            "Avere strumenti di critica del modello significa che a volte ti "
            "dicono cose scomode. Tre osservazioni.\n\n"
            "Uno: Posterior Predictive Check su 2 navi rappresentative. 4 "
            "statistiche sintetiche — media e std di speed, autocorrelazione del "
            "turn rate, frazione di valori estremi — confrontate fra simulazioni "
            "dal posterior e dati osservati. Modello adeguato sulle prime tre, "
            "qualche discrepanza sul tail behavior. Niente è perfetto, ma niente "
            "è nascosto.\n\n"
            "Due: Prior Sensitivity Analysis. Abbiamo rifittato il BMM sotto tre "
            "specificazioni alternative — loose, default, tight. La correlazione "
            "di Spearman fra i ranking di sospetto-score delle navi: min ρ = "
            "meno uno (loose verso default). Verdetto onesto: il modello non è "
            "robusto al prior sotto la soglia standard di 0.85. Le conclusioni "
            "dipendono dalle scelte di prior. Questo è esattamente quello che la "
            "PSA serve a rivelare, non un difetto da nascondere.\n\n"
            "Tre: bootstrap CI clustered per MMSI, 1000 iterazioni. L'intervallo "
            "di confidenza al 95 percento sul PR-AUC test include lo zero — "
            "coerente con la bassa stima puntuale."
        ),
    },
    # ---- Slide 5: Onestà & next ----
    {
        "kind": "content_dark",
        "title": "Onestà e next step",
        "subtitle": "Cosa dichiariamo prima della Q&A",
        "image": None,  # full text slide su dark bg
        "bullets": [
            "Dati simulati — segnale by design, numeri = upper bound metodologico",
            "Prior sensitivity rivela instabilità reale → NUTS o più pilot vessel",
            "Next step: AIS reali (AISHub / Spire), 1000+ navi, ri-validazione",
            "Pipeline production-ready come scheletro. Mancano i dati per riempirla.",
        ],
        "notes": (
            "Tre cose vanno dichiarate prima della Q&A.\n\n"
            "Uno: i dati sono simulati. Un simulatore con dark ratio configurabile "
            "al 50 percento e blackout uniformi nel tempo. L'audit del simulatore "
            "conferma che il segnale è presente per costruzione: drop SOG del 60 "
            "percento nelle 2 ore pre-blackout. I numeri di questa presentazione "
            "sono un upper bound metodologico, non una promessa di deployment.\n\n"
            "Due: la prior sensitivity rivela un'instabilità reale del modello. "
            "La metodologia ha funzionato — ha individuato il problema. Strada "
            "concreta: gerarchia Bayesiana completa, sampling NUTS, invece di "
            "Empirical Bayes a due stadi. Oppure aumentare il numero di navi "
            "pilota per stabilizzare l'EB.\n\n"
            "Tre: next step naturale è l'ingestione di AIS reali — AISHub o Spire "
            "— con più navi, idealmente 1000 o più, e ri-validazione end-to-end. "
            "Questa pipeline è production-ready come scheletro metodologico. "
            "Quello che manca sono i dati per riempirlo.\n\n"
            "Grazie."
        ),
    },
]


# ============================================================================
# Layout helpers
# ============================================================================
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def _set_bg(slide, color):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    # Push to back
    spTree = bg._element.getparent()
    spTree.remove(bg._element)
    spTree.insert(2, bg._element)


def _add_text(slide, text, *, x, y, w, h, font=FONT_BODY, size=14,
              color=COLOR_TEXT_DARK, bold=False, italic=False,
              align=PP_ALIGN.LEFT):
    tx = slide.shapes.add_textbox(x, y, w, h)
    tf = tx.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0)
    tf.margin_top = tf.margin_bottom = Inches(0)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return tx


def _add_bullets(slide, bullets, *, x, y, w, h, color=COLOR_TEXT_DARK,
                 size=18, font=FONT_BODY):
    tx = slide.shapes.add_textbox(x, y, w, h)
    tf = tx.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0)
    tf.margin_top = tf.margin_bottom = Inches(0)
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(10)
        # bullet using actual unicode marker (controllato, no double-bullets)
        r1 = p.add_run()
        r1.text = "›  "
        r1.font.name = font
        r1.font.size = Pt(size)
        r1.font.bold = True
        r1.font.color.rgb = COLOR_TEAL
        r2 = p.add_run()
        r2.text = b
        r2.font.name = font
        r2.font.size = Pt(size)
        r2.font.color.rgb = color
    return tx


def _set_notes(slide, notes_text):
    notes = slide.notes_slide.notes_text_frame
    notes.text = notes_text


# ============================================================================
# Build slides
# ============================================================================
def build_slide_content_light(prs, spec):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_bg(slide, COLOR_BG_LIGHT)

    # Title block (top left)
    _add_text(slide, spec["title"],
              x=Inches(0.6), y=Inches(0.4), w=Inches(12.0), h=Inches(0.8),
              font=FONT_TITLE, size=38, bold=True, color=COLOR_MIDNIGHT)
    _add_text(slide, spec["subtitle"],
              x=Inches(0.6), y=Inches(1.15), w=Inches(12.0), h=Inches(0.45),
              font=FONT_BODY, size=18, italic=True, color=COLOR_TEAL)

    # Image (right column) + bullets (left column)
    if spec.get("image") and Path(spec["image"]).exists():
        img_w = Inches(6.6)
        img_h = Inches(5.0)
        img_x = SLIDE_W - img_w - Inches(0.5)
        img_y = Inches(1.85)
        slide.shapes.add_picture(spec["image"], img_x, img_y,
                                 width=img_w, height=img_h)
        bullets_w = Inches(6.0)
    else:
        bullets_w = Inches(12.0)

    _add_bullets(slide, spec["bullets"],
                 x=Inches(0.6), y=Inches(2.0), w=bullets_w, h=Inches(4.5),
                 size=17, color=COLOR_TEXT_DARK)

    # Footer
    _add_text(slide, "AIS Dark Fleet Predictor — contest 2026",
              x=Inches(0.6), y=Inches(7.05), w=Inches(8.0), h=Inches(0.3),
              size=10, color=COLOR_TEXT_MUTE)

    _set_notes(slide, spec["notes"])
    return slide


def build_slide_content_dark(prs, spec):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(slide, COLOR_BG_DARK)

    # Title block
    _add_text(slide, spec["title"],
              x=Inches(0.6), y=Inches(0.4), w=Inches(12.0), h=Inches(0.9),
              font=FONT_TITLE, size=44, bold=True, color=COLOR_WHITE)
    _add_text(slide, spec["subtitle"],
              x=Inches(0.6), y=Inches(1.25), w=Inches(12.0), h=Inches(0.5),
              font=FONT_BODY, size=20, italic=True,
              color=RGBColor(0xCA, 0xDC, 0xFC))  # ice blue

    if spec.get("image") and Path(spec["image"]).exists():
        img_w = Inches(6.6)
        img_h = Inches(5.0)
        img_x = SLIDE_W - img_w - Inches(0.5)
        img_y = Inches(1.95)
        slide.shapes.add_picture(spec["image"], img_x, img_y,
                                 width=img_w, height=img_h)
        bullets_w = Inches(6.0)
    else:
        bullets_w = Inches(12.0)

    _add_bullets(slide, spec["bullets"],
                 x=Inches(0.6), y=Inches(2.2), w=bullets_w, h=Inches(4.5),
                 size=18, color=COLOR_WHITE)

    _add_text(slide, "AIS Dark Fleet Predictor — contest 2026",
              x=Inches(0.6), y=Inches(7.05), w=Inches(8.0), h=Inches(0.3),
              size=10, color=RGBColor(0x90, 0xA4, 0xBF))

    _set_notes(slide, spec["notes"])
    return slide


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    for spec in SLIDES:
        if spec["kind"] == "content_dark":
            build_slide_content_dark(prs, spec)
        else:
            build_slide_content_light(prs, spec)

    prs.save(str(OUT_PATH))
    print(f"OK: {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.1f} kB)")


if __name__ == "__main__":
    main()
