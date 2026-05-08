"""
ui/app.py — CogniMed
AI cognition applied to medicine.
"""

import os
import sys
import logging
import random
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import RESULTS_DIR, MEDICAL_DISCLAIMER
except ImportError:
    RESULTS_DIR = str(_REPO_ROOT / "results")
    MEDICAL_DISCLAIMER = (
        "For research and educational use only. Not a substitute for professional medical advice."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_engine = None
_engine_error = None


def _get_engine():
    global _engine, _engine_error
    if _engine is not None:
        return _engine, None
    if _engine_error is not None:
        return None, _engine_error
    try:
        from lora_inference import LoRAInferenceEngine
        eng = LoRAInferenceEngine()
        eng.load()
        _engine = eng
        return _engine, None
    except FileNotFoundError as e:
        _engine_error = str(e)
        return None, _engine_error
    except Exception as e:
        _engine_error = str(e)
        logger.warning(f"Engine load failed: {e}", exc_info=True)
        return None, _engine_error


_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    try:
        from lora_inference import FAISSRetriever
        r = FAISSRetriever()
        r.load()
        _retriever = r
    except Exception as e:
        logger.warning(f"FAISS retriever unavailable ({e}).")
        _retriever = None
    return _retriever


_MOCK_CHUNKS = [
    {"rank": 1, "chunk": "A meta-analysis of 42 randomized controlled trials found a pooled effect size of 0.68 (95% CI: 0.54-0.82), supporting clinical benefit of the intervention.", "score": 0.91, "metadata": {"pubmed_id": "PMC7654321"}},
    {"rank": 2, "chunk": "Observational cohort data (n=18,000 patients followed for 5 years) showed a 23% reduction in adverse events compared with standard care (p<0.001).", "score": 0.78, "metadata": {"pubmed_id": "PMC5432167"}},
    {"rank": 3, "chunk": "Systematic review of phase III trials concluded the treatment was well-tolerated with an acceptable safety profile across patient subgroups.", "score": 0.65, "metadata": {"pubmed_id": "PMC9012345"}},
]

_MOCK_ANSWERS = {
    "yes": "Multiple randomized controlled trials and meta-analyses support this conclusion with statistically significant benefit and an acceptable safety profile.",
    "no":  "Current literature does not support a significant clinical benefit. Methodological limitations across primary studies further reduce confidence.",
}


def _verdict_pill(label: str) -> str:
    cls   = {"YES": "cm-yes", "NO": "cm-no"}.get(label, "cm-uncertain")
    glyph = {"YES": "✓", "NO": "✗"}.get(label, "?")
    text  = {"YES": "YES", "NO": "NO"}.get(label, "UNCERTAIN")
    return (
        "<div class='cm-verdict'>"
        "<div class='cm-verdict-label'>Predicted answer</div>"
        f"<div class='cm-verdict-pill {cls}'>"
        f"<span class='cm-glyph'>{glyph}</span>"
        f"<span class='cm-verdict-text'>{text}</span>"
        "</div>"
        "</div>"
    )


def _evidence_html(chunks, demo: bool = False) -> str:
    if not chunks:
        return (
            "<div class='cm-empty'>"
            "<div class='cm-empty-icon'>📭</div>"
            "<div>No supporting passages were retrieved for this question.</div>"
            "</div>"
        )

    title_tag = " · demo" if demo else ""
    parts = [
        "<div class='cm-evidence-wrap'>",
        f"<div class='cm-section-title'>📚 Supporting evidence ({len(chunks)} passages{title_tag})</div>",
    ]
    for c in chunks:
        score = float(c.get("score", 0.0))
        pid   = c.get("metadata", {}).get("pubmed_id", "—")
        rank  = c.get("rank", "?")
        text  = (c.get("chunk") or "").strip()
        if len(text) > 360:
            text = text[:360].rstrip() + "…"

        if   score > 0.75: tone = "cm-tone-strong"
        elif score > 0.55: tone = "cm-tone-medium"
        else:              tone = "cm-tone-soft"

        parts.append(
            "<div class='cm-card " + tone + "'>"
            "<div class='cm-card-head'>"
            f"<span class='cm-card-rank'>Rank #{rank}</span>"
            f"<span class='cm-card-meta'>PubMed ID · {pid}  ·  similarity {score:.3f}</span>"
            "</div>"
            f"<div class='cm-card-text'>{text}</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _answer_markdown(label: str, chunks, avg_score: float) -> str:
    verdict = {
        "YES":       "the answer is **YES**.",
        "NO":        "the answer is **NO**.",
        "UNCERTAIN": "the model could not produce a confident yes / no answer.",
    }[label]

    if not chunks:
        return f"### Reasoning\n\n{verdict.capitalize()}"

    top = chunks[0]
    pid = top.get("metadata", {}).get("pubmed_id", "—")
    snippet = (top.get("chunk") or "").strip()
    if len(snippet) > 320:
        snippet = snippet[:320].rstrip() + "…"

    return (
        "### Reasoning\n\n"
        f"Based on **{len(chunks)} retrieved PubMedQA passages** "
        f"(average similarity *{avg_score:.3f}*), {verdict}\n\n"
        f"**Top supporting passage** — PubMed ID `{pid}` (similarity {top.get('score', 0):.3f}):\n\n"
        f"> {snippet}\n\n"
        "_See the full evidence panel below for all retrieved passages._"
    )


def _mock_query(question, top_k):
    time.sleep(0.4)
    key   = random.choice(["yes", "no"])
    label = key.upper()
    chunks = _MOCK_CHUNKS[:top_k]
    avg = sum(c["score"] for c in chunks) / len(chunks)

    pill   = _verdict_pill(label)
    answer = "### Reasoning\n\n" + _MOCK_ANSWERS[key] + (
        f"\n\n_Demo mode — based on {len(chunks)} simulated passages "
        f"(avg similarity {avg:.3f})._"
    )
    ev = _evidence_html(chunks, demo=True)
    return answer, pill, ev


def answer_medical_question(question, top_k):
    if not (question or "").strip():
        return (
            "_Please enter a medical question above to begin._",
            "<div class='cm-verdict-empty'>Awaiting question…</div>",
            "",
        )

    top_k = int(top_k)

    engine, err = _get_engine()
    if engine is None:
        ans, pill, ev = _mock_query(question, top_k)
        prefix = (
            f"> ⚠️ **Model not loaded** — falling back to demo. Reason: `{err}`\n\n"
        )
        return prefix + ans, pill, ev

    retriever = _get_retriever()
    chunks    = []
    avg_score = 0.0
    if retriever:
        try:
            chunks = retriever.retrieve(question, top_k=top_k)
            if chunks:
                avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
            for i, c in enumerate(chunks):
                c["rank"] = i + 1
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")

    try:
        result = engine.answer(question, context_chunks=chunks or None)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        ans, pill, ev = _mock_query(question, top_k)
        prefix = f"> ❌ **Inference error:** `{e}`\n\n"
        return prefix + ans, pill, ev

    label = result.get("predicted_label", "UNCERTAIN").upper()
    pill  = _verdict_pill(label)
    ans   = _answer_markdown(label, chunks, avg_score)

    if chunks:
        ev = _evidence_html(chunks)
    else:
        ev = (
            "<div class='cm-empty'>"
            "<div class='cm-empty-icon'>🔍</div>"
            "<div>No FAISS index available — answer drawn from the model's parametric knowledge.</div>"
            "</div>"
        )
    return ans, pill, ev


CSS = """
:root, .gradio-container, .dark {
    --background-fill-primary: #ECDFC8 !important;
    --background-fill-secondary: #F6ECD6 !important;
    --body-background-fill: #ECDFC8 !important;
    --color-accent: #8C6A3F !important;
    --color-accent-soft: #C4956C !important;
    --button-primary-background-fill: linear-gradient(135deg,#8C6A3F 0%,#6E5230 100%) !important;
    --button-primary-text-color: #FAF1E1 !important;
    --body-text-color: #3D2E1F !important;
    --border-color-primary: #D9C4A1 !important;
    --block-background-fill: #F8F1DF !important;
    --block-border-color: #D9C4A1 !important;
    --input-background-fill: #FAF6EA !important;
    --input-border-color: #D9C4A1 !important;
    --neutral-50: #FAF6EA !important;
    --neutral-100: #F6ECD6 !important;
    --neutral-200: #ECDFC8 !important;
    --primary-50:  #F4E9D2 !important;
    --primary-100: #E4D0AC !important;
    --primary-200: #D9C4A1 !important;
    --primary-300: #C4956C !important;
    --primary-400: #A07A4F !important;
    --primary-500: #8C6A3F !important;
    --primary-600: #7A5A33 !important;
    --primary-700: #6E5230 !important;
}

html, body, .gradio-container, .main, gradio-app {
    background: linear-gradient(135deg, #ECDFC8 0%, #DCC9A8 100%) !important;
    color: #3D2E1F !important;
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
}

footer { display: none !important; }

.cm-hero {
    background: linear-gradient(135deg, #C49363 0%, #8C6A3F 60%, #6E5230 100%);
    color: #FAF1E1 !important;
    padding: 56px 32px;
    border-radius: 22px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 14px 40px rgba(80, 55, 30, 0.18);
    position: relative;
    overflow: hidden;
}
.cm-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(255,242,220,0.18), transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(255,242,220,0.10), transparent 45%);
    pointer-events: none;
}
.cm-hero-mark {
    font-size: 3.6rem;
    line-height: 1;
    margin-bottom: 6px;
    filter: drop-shadow(0 3px 6px rgba(50,30,15,0.3));
}
.cm-hero h1 {
    font-size: 3.3rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 4px 0 6px !important;
    color: #FFF6E1 !important;
}
.cm-hero p.tagline {
    font-size: 1.15rem;
    margin: 0 !important;
    color: #FFF6E1 !important;
    opacity: 0.92;
    font-style: italic;
    letter-spacing: 0.4px;
}

.tabs > .tab-nav {
    background: #F4E9D2 !important;
    border-radius: 14px !important;
    border: 1px solid #D9C4A1 !important;
    padding: 4px !important;
}
.tabs > .tab-nav button {
    color: #6E5230 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    background: transparent !important;
}
.tabs > .tab-nav button.selected {
    background: linear-gradient(135deg,#C49363,#8C6A3F) !important;
    color: #FAF1E1 !important;
    box-shadow: 0 4px 12px rgba(80,55,30,0.22) !important;
}

.gradio-container .block {
    background: #F8F1DF !important;
    border: 1px solid #D9C4A1 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 14px rgba(80, 55, 30, 0.05);
}
.gradio-container label {
    color: #6E5230 !important;
    font-weight: 600 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .gr-textbox textarea,
.gradio-container .gr-textbox input {
    background: #FFFBF1 !important;
    color: #3D2E1F !important;
    border: 1.5px solid #D9C4A1 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #8C6A3F !important;
    box-shadow: 0 0 0 3px rgba(140, 106, 63, 0.18) !important;
    outline: none !important;
}

/* Slider — beige/brown */
.gradio-container input[type="range"] {
    accent-color: #8C6A3F !important;
}
.gradio-container [data-testid="slider"],
.gradio-container .gradio-slider {
    --primary-50:  #F4E9D2 !important;
    --primary-100: #E4D0AC !important;
    --primary-200: #D9C4A1 !important;
    --primary-300: #C4956C !important;
    --primary-400: #A07A4F !important;
    --primary-500: #8C6A3F !important;
    --primary-600: #7A5A33 !important;
    --primary-700: #6E5230 !important;
}
.gradio-container [data-testid="slider"] input[type="number"],
.gradio-container .gradio-slider input[type="number"] {
    background: #FFFBF1 !important;
    border: 1.5px solid #D9C4A1 !important;
    border-radius: 10px !important;
    color: #3D2E1F !important;
}
.gradio-container [data-testid="slider"] label,
.gradio-container .gradio-slider label {
    background: transparent !important;
    color: #6E5230 !important;
}

.gradio-container button.primary,
.gradio-container .primary {
    background: linear-gradient(135deg, #8C6A3F 0%, #6E5230 100%) !important;
    color: #FAF1E1 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 14px rgba(80, 55, 30, 0.18);
    transition: transform 0.12s ease;
}
.gradio-container button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(80, 55, 30, 0.26);
}
.gradio-container button.secondary {
    background: #F4E9D2 !important;
    color: #6E5230 !important;
    border: 1.5px solid #D9C4A1 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

.cm-verdict {
    text-align: center;
    padding: 28px 18px;
    background: linear-gradient(135deg, #FBF4E0 0%, #F0E3C7 100%);
    border: 1px solid #D9C4A1;
    border-radius: 22px;
    margin: 12px 0 22px;
    box-shadow: 0 8px 24px rgba(80, 55, 30, 0.07);
}
.cm-verdict-label {
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #8C6A3F;
    margin-bottom: 14px;
}
.cm-verdict-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 14px 38px;
    border-radius: 999px;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 2px;
    box-shadow: 0 6px 18px rgba(80, 55, 30, 0.15),
                inset 0 1px 0 rgba(255,255,255,0.5);
    transition: transform 0.18s ease;
}
.cm-verdict-pill:hover { transform: translateY(-1px) scale(1.02); }
.cm-glyph {
    display: inline-flex;
    width: 32px; height: 32px;
    align-items: center; justify-content: center;
    border-radius: 50%;
    font-size: 1.1rem;
    font-weight: 900;
}
.cm-yes {
    background: linear-gradient(135deg, #E2EDD2 0%, #C5D6A8 100%);
    color: #3F5A2D;
    border: 2.5px solid #8AA86C;
}
.cm-yes .cm-glyph { background:#8AA86C; color:#FFF; }
.cm-no {
    background: linear-gradient(135deg, #F5DBCE 0%, #E8B7A2 100%);
    color: #6F2D1A;
    border: 2.5px solid #C26A52;
}
.cm-no .cm-glyph { background:#C26A52; color:#FFF; }
.cm-uncertain {
    background: linear-gradient(135deg, #F1E4C9 0%, #E2CFA0 100%);
    color: #6E5230;
    border: 2.5px solid #B89154;
}
.cm-uncertain .cm-glyph { background:#B89154; color:#FFF; }
.cm-verdict-empty {
    text-align: center;
    padding: 36px;
    background: #F8F1DF;
    border: 1px dashed #D9C4A1;
    border-radius: 18px;
    color: #8C6A3F;
    font-style: italic;
}

.cm-section-title {
    color: #6E5230;
    font-size: 1.08rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin: 18px 0 12px;
}

/* Reasoning markdown card — single layer, no nested boxes */
.cm-answer-md {
    background: #F8F1DF !important;
    border: 1px solid #D9C4A1 !important;
    border-radius: 16px !important;
    padding: 24px 30px !important;
    box-shadow: 0 4px 14px rgba(80, 55, 30, 0.05) !important;
    color: #3D2E1F !important;
    overflow: hidden !important;
}
.cm-answer-md > *,
.cm-answer-md > div,
.cm-answer-md .prose,
.cm-answer-md .markdown {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gradio-container .prose,
.gradio-container .markdown { color: #3D2E1F !important; }
.gradio-container .prose blockquote,
.gradio-container .markdown blockquote {
    border-left: 4px solid #C4956C !important;
    background: #FFFBF1 !important;
    color: #5C4628 !important;
    padding: 12px 18px !important;
    margin: 14px 0 !important;
    border-radius: 0 12px 12px 0 !important;
    font-style: italic;
    line-height: 1.55;
}
.gradio-container .prose h3,
.gradio-container .markdown h3 {
    color: #6E5230 !important;
    border-bottom: 2px solid #D9C4A1 !important;
    padding-bottom: 8px !important;
    margin-top: 4px !important;
    margin-bottom: 14px !important;
    font-size: 1.15rem !important;
}

.cm-evidence-wrap { margin-top: 8px; }
.cm-card {
    background: #FFFBF1;
    border: 1px solid #E4D0AC;
    border-left: 5px solid #8C6A3F;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0;
    box-shadow: 0 3px 10px rgba(80, 55, 30, 0.05);
}
.cm-card.cm-tone-strong  { border-left-color: #6B8E5A; background:#F4F0DD; }
.cm-card.cm-tone-medium  { border-left-color: #B89154; background:#FBF3DD; }
.cm-card.cm-tone-soft    { border-left-color: #C4956C; background:#FFF7E6; }

.cm-card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 6px;
}
.cm-card-rank { font-weight: 700; color: #6E5230; font-size: 0.95rem; }
.cm-card-meta { color: #8C6A3F; font-size: 0.82rem; letter-spacing: 0.3px; }
.cm-card-text { color: #3D2E1F; line-height: 1.55; font-size: 0.95rem; }

.cm-empty {
    text-align: center;
    padding: 28px 16px;
    background: #F4E9D2;
    border: 1px dashed #D9C4A1;
    border-radius: 14px;
    color: #6E5230;
}
.cm-empty-icon { font-size: 2.2rem; margin-bottom: 8px; }

.gradio-container .accordion {
    background: #F4E9D2 !important;
    border: 1px solid #D9C4A1 !important;
    border-radius: 14px !important;
}
.gradio-container .accordion summary,
.gradio-container .accordion .label-wrap {
    color: #6E5230 !important;
    font-weight: 700 !important;
}
.gradio-container .accordion .icon,
.gradio-container .accordion .label-wrap > svg,
.gradio-container .accordion summary::-webkit-details-marker,
.gradio-container .accordion summary > svg {
    display: none !important;
}
.gradio-container .accordion .label-wrap::after {
    content: "▾";
    margin-left: auto;
    color: #8C6A3F;
    font-size: 1rem;
    transition: transform 0.18s ease;
    transform: rotate(0deg) !important;
}
.gradio-container .accordion.open .label-wrap::after,
.gradio-container .accordion[open] .label-wrap::after {
    transform: rotate(180deg) !important;
}

/* Dropdown — flat, single border */
.gradio-container [data-testid="dropdown"],
.gradio-container .gr-dropdown {
    background: #FFFBF1 !important;
    border: 1.5px solid #D9C4A1 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    padding: 2px !important;
}
.gradio-container [data-testid="dropdown"] *,
.gradio-container .gr-dropdown * {
    box-shadow: none !important;
    outline: none !important;
}
.gradio-container [data-testid="dropdown"] input,
.gradio-container [data-testid="dropdown"] .wrap-inner,
.gradio-container [data-testid="dropdown"] .secondary-wrap,
.gradio-container [data-testid="dropdown"] .single-select,
.gradio-container [data-testid="dropdown"] .selected,
.gradio-container [data-testid="dropdown"] .token,
.gradio-container [data-testid="dropdown"] .token-remove {
    background: transparent !important;
    border: none !important;
    color: #3D2E1F !important;
}
.gradio-container [data-testid="dropdown"]:focus-within {
    border-color: #8C6A3F !important;
    box-shadow: 0 0 0 3px rgba(140, 106, 63, 0.18) !important;
}

.cm-disclaimer {
    text-align: center;
    color: #8C6A3F;
    font-size: 0.85rem;
    font-style: italic;
    margin-top: 28px;
    padding: 14px;
    background: #F4E9D2;
    border: 1px dashed #D9C4A1;
    border-radius: 12px;
}

.tabitem { padding-top: 16px !important; }

.cm-about { padding: 8px 4px; }
.cm-about h2 {
    color: #6E5230 !important;
    font-size: 1.5rem;
    margin-top: 18px;
    margin-bottom: 10px;
}
.cm-about h3 {
    color: #8C6A3F !important;
    font-size: 1.1rem;
    margin-top: 16px;
    margin-bottom: 8px;
}
.cm-about p, .cm-about li { color: #3D2E1F; line-height: 1.6; }
.cm-about .cm-credit {
    background: #F4E9D2;
    border: 1px solid #D9C4A1;
    border-left: 4px solid #8C6A3F;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 14px;
    color: #5C4628;
}
.cm-arch {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    justify-content: center;
    gap: 10px;
    margin: 16px 0 8px;
}
.cm-arch-step {
    flex: 1 1 130px;
    min-width: 130px;
    max-width: 170px;
    background: linear-gradient(135deg, #FFFBF1 0%, #F4E9D2 100%);
    border: 1.5px solid #D9C4A1;
    border-top: 4px solid #8C6A3F;
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(80,55,30,0.06);
}
.cm-arch-step .cm-arch-icon { font-size: 2rem; margin-bottom: 4px; display:block; }
.cm-arch-step .cm-arch-name { font-weight: 700; color: #6E5230; font-size: 0.95rem; }
.cm-arch-step .cm-arch-sub  { font-size: 0.78rem; color: #8C6A3F; margin-top: 4px; }
.cm-arch-arrow {
    align-self: center;
    color: #8C6A3F;
    font-size: 1.6rem;
    font-weight: 800;
    flex: 0 0 auto;
}

.cm-ask-card {
    background: linear-gradient(135deg, #F8F1DF 0%, #F4E9D2 100%);
    border: 1.5px solid #D9C4A1;
    border-radius: 18px;
    padding: 22px 24px;
    margin: 6px 0 18px;
    box-shadow: 0 6px 18px rgba(80, 55, 30, 0.06);
}
.cm-ask-card textarea {
    font-size: 1.05rem !important;
    padding: 14px 16px !important;
}
"""


SAMPLE_QUESTIONS = [
    "Does metformin reduce cardiovascular mortality in type 2 diabetes patients?",
    "Is cognitive behavioral therapy effective for treating major depressive disorder?",
    "Do statins reduce the risk of stroke in patients with high cholesterol?",
    "Can aspirin therapy prevent recurrent myocardial infarction?",
    "Is remdesivir effective in reducing mortality from COVID-19?",
    "Does early enteral nutrition improve outcomes in critically ill patients?",
    "Is mindfulness-based stress reduction effective for chronic pain?",
    "Do ACE inhibitors reduce mortality in patients with heart failure?",
]


def _about_html() -> str:
    return (
        "<div class='cm-about'>"
        "<h2>About CogniMed</h2>"
        "<p><em>AI cognition applied to medicine.</em></p>"
        "<p>CogniMed is a Retrieval-Augmented Generation system for biomedical "
        "question answering. It combines a domain-tuned sentence-transformer for "
        "evidence retrieval with a LoRA-fine-tuned BioGPT-Large language model for "
        "binary yes / no clinical reasoning, grounded in PubMed-derived passages.</p>"

        "<h3>Architecture</h3>"
        "<div class='cm-arch'>"
        "<div class='cm-arch-step'>"
        "<span class='cm-arch-icon'>🩺</span>"
        "<div class='cm-arch-name'>Question</div>"
        "<div class='cm-arch-sub'>Clinical query</div>"
        "</div>"
        "<div class='cm-arch-arrow'>→</div>"
        "<div class='cm-arch-step'>"
        "<span class='cm-arch-icon'>🔤</span>"
        "<div class='cm-arch-name'>Encoder</div>"
        "<div class='cm-arch-sub'>S-PubMedBERT</div>"
        "</div>"
        "<div class='cm-arch-arrow'>→</div>"
        "<div class='cm-arch-step'>"
        "<span class='cm-arch-icon'>📚</span>"
        "<div class='cm-arch-name'>Retrieval</div>"
        "<div class='cm-arch-sub'>FAISS over PubMedQA</div>"
        "</div>"
        "<div class='cm-arch-arrow'>→</div>"
        "<div class='cm-arch-step'>"
        "<span class='cm-arch-icon'>🧠</span>"
        "<div class='cm-arch-name'>Reasoner</div>"
        "<div class='cm-arch-sub'>BioGPT-Large + LoRA</div>"
        "</div>"
        "<div class='cm-arch-arrow'>→</div>"
        "<div class='cm-arch-step'>"
        "<span class='cm-arch-icon'>✅</span>"
        "<div class='cm-arch-name'>Verdict</div>"
        "<div class='cm-arch-sub'>Yes / No + evidence</div>"
        "</div>"
        "</div>"

        "<h3>Components</h3>"
        "<ul>"
        "<li><strong>Retriever:</strong> S-PubMedBert-MS-MARCO sentence-transformer + FAISS Inner-Product index over chunked PubMedQA abstracts.</li>"
        "<li><strong>Reasoner:</strong> microsoft/BioGPT-Large-PubMedQA fine-tuned with LoRA (rank 32, alpha 64) on the binary subset of PubMedQA.</li>"
        "<li><strong>Decoding:</strong> Greedy first-token decoding for stable yes / no classification.</li>"
        "<li><strong>Grounding:</strong> The verdict is presented alongside the top retrieved PubMedQA passages so reasoning is traceable.</li>"
        "</ul>"

        "<h3>Created by</h3>"
        "<div class='cm-credit'>"
        "<strong>Hrishita Panjetha</strong>"
        "</div>"
        "</div>"
    )


def build_ui():
    import gradio as gr

    with gr.Blocks(
        title="CogniMed — AI cognition applied to medicine",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="stone",
            secondary_hue="stone",
            neutral_hue="stone",
        ),
    ) as demo:

        gr.HTML(
            "<div class='cm-hero'>"
            "<div class='cm-hero-mark'>🧠⚕️</div>"
            "<h1>CogniMed</h1>"
            "<p class='tagline'>AI cognition applied to medicine</p>"
            "</div>"
        )

        with gr.Tabs():

            with gr.Tab("Ask"):
                with gr.Column(elem_classes=["cm-ask-card"]):
                    gr.HTML("<div class='cm-section-title'>🩺 Ask a medical question</div>")
                    question_input = gr.Textbox(
                        placeholder="e.g.  Does aspirin reduce cardiovascular risk in adults over 50?",
                        lines=3,
                        max_lines=8,
                        show_label=False,
                    )

                    with gr.Row():
                        top_k_slider = gr.Slider(
                            1, 10, value=5, step=1,
                            label="Evidence passages to retrieve",
                        )

                    with gr.Row():
                        submit_btn = gr.Button("Analyze", variant="primary", size="lg")
                        clear_btn  = gr.Button("Clear",   variant="secondary", size="lg")

                    with gr.Accordion("💡 Browse sample questions", open=False):
                        sample_dd = gr.Dropdown(
                            choices=SAMPLE_QUESTIONS,
                            label="Pick a sample",
                            show_label=False,
                            interactive=True,
                            container=False,
                        )

                gr.HTML("<div class='cm-section-title'>📋 Result</div>")
                verdict_html = gr.HTML(
                    "<div class='cm-verdict-empty'>Awaiting your question…</div>"
                )
                answer_md = gr.Markdown(
                    "_Ask a medical question above and CogniMed will respond with a "
                    "yes / no verdict grounded in retrieved PubMedQA evidence._",
                    elem_classes=["cm-answer-md"],
                )
                evidence_html = gr.HTML("")

                inputs  = [question_input, top_k_slider]
                outputs = [answer_md, verdict_html, evidence_html]

                submit_btn.click(fn=answer_medical_question, inputs=inputs, outputs=outputs)
                question_input.submit(fn=answer_medical_question, inputs=inputs, outputs=outputs)
                clear_btn.click(
                    fn=lambda: (
                        "",
                        "_Ask a medical question above and CogniMed will respond with a "
                        "yes / no verdict grounded in retrieved PubMedQA evidence._",
                        "<div class='cm-verdict-empty'>Awaiting your question…</div>",
                        "",
                    ),
                    outputs=[question_input] + outputs,
                )
                sample_dd.change(fn=lambda x: x or "", inputs=sample_dd, outputs=question_input)

            with gr.Tab("About"):
                gr.HTML(_about_html())

        gr.HTML(
            "<div class='cm-disclaimer'>"
            f"CogniMed · {MEDICAL_DISCLAIMER}"
            "</div>"
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()