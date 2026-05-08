cd "/teamspace/studios/this_studio/medical-diagnosis-llm-main 4"

cat > README.md <<'COGNIMED_README_END'
---
title: CogniMed
emoji: 🧠
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: "5.13.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🧠 CogniMed

### *AI cognition applied to medicine.*

**Evidence-grounded biomedical question answering** powered by retrieval-augmented generation, a domain-tuned sentence-transformer, and a LoRA-fine-tuned BioGPT-Large reasoner.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-CogniMed-FFD21E?style=for-the-badge&labelColor=8C6A3F)](https://huggingface.co/spaces/Hrishita-P/cognimed)
[![Built with Gradio](https://img.shields.io/badge/UI-Gradio%205-orange?style=for-the-badge)](https://gradio.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

[**Try the live demo →**](https://huggingface.co/spaces/Hrishita-P/cognimed)

</div>

---

## 📖 Overview

CogniMed answers binary medical questions (*yes / no*) by retrieving peer-reviewed evidence from the **PubMedQA** corpus and grounding the response in those passages. It combines two complementary models:

- A **domain-tuned sentence-transformer** ([pritamdeka/S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO)) for evidence retrieval over a FAISS index of chunked PubMed abstracts.
- A **LoRA-fine-tuned [microsoft/BioGPT-Large-PubMedQA](https://huggingface.co/microsoft/BioGPT-Large-PubMedQA)** as the binary clinical reasoner.

The verdict is presented alongside the supporting passages, so reasoning is **traceable** — every claim points to a PubMed ID.

## 🏗️ Architecture

Question → S-PubMedBERT encoder → FAISS retrieval over PubMedQA → BioGPT-Large + LoRA reasoner → Verdict (✓ YES / ✗ NO / ❓ UNCERTAIN) + supporting passages.

## ✨ Features

- 🩺 Binary yes / no clinical classification with evidence grounding
- 📚 Retrieval-augmented context from chunked PubMedQA abstracts
- 🧠 LoRA-fine-tuned BioGPT-Large (rank 32, alpha 64) on the binary subset of PubMedQA
- 🎯 Confidence-margin abstention — returns UNCERTAIN on borderline predictions instead of forcing a low-confidence label
- 🎨 Beige-themed Gradio UI with verdict pills, evidence cards, and an architecture overview
- 🔗 PubMed ID traceability — every retrieved passage links back to its source

## 🚀 Live Demo

Try it instantly without installing anything:

**👉 [huggingface.co/spaces/Hrishita-P/cognimed](https://huggingface.co/spaces/Hrishita-P/cognimed)**

First query takes ~3–5 minutes (Space cold-start downloads the base model). Subsequent queries run in 30–60 seconds on the free CPU tier.

## 🛠️ Run Locally

### Prerequisites

- Python 3.11+
- ~10 GB free disk space (for the BioGPT-Large base model cache)
- Optional: CUDA GPU for fast inference (CPU works but slow)

### Setup

    git clone https://github.com/hrishitapanjetha/cognimed.git
    cd cognimed
    pip install -r requirements.txt

### Launch the UI

    python app.py

This opens the Gradio app at http://localhost:7860. The first query downloads microsoft/biogpt-large (~6 GB) and the embedder (~440 MB) into the Hugging Face cache.

### CLI inference

    python run.py query "Does aspirin reduce the risk of recurrent myocardial infarction?"

### Rebuild the FAISS index from source data

If you've changed the embedder or the corpus:

    python scripts/rebuild_index.py

## 🧪 Sample Questions

Try these to see CogniMed's range:

| Question | Expected |
|----------|----------|
| Does aspirin reduce the risk of recurrent myocardial infarction? | ✅ YES |
| Does cognitive behavioral therapy reduce symptoms in major depressive disorder? | ✅ YES |
| Does the Mediterranean diet reduce cardiovascular events in high-risk adults? | ✅ YES |
| Does folic acid supplementation before conception reduce neural tube defects? | ✅ YES |
| Is bed rest effective for treating acute lower back pain? | ❌ NO |
| Is high-dose vitamin C effective as a cancer treatment? | ❌ NO |
| Does ivermectin reduce mortality in COVID-19 patients? | ❌ NO |
| Does multivitamin supplementation reduce all-cause mortality in healthy adults? | ❌ NO |

## ⚙️ Technical Details

### Model

| Component | Value |
|-----------|-------|
| Base model | microsoft/BioGPT-Large-PubMedQA |
| Adapter | PEFT LoRA, rank 32, alpha 64 |
| Target modules | q_proj, k_proj, v_proj, out_proj |
| Training data | PubMedQA, binary yes/no subset (maybe records filtered out) |
| Training objective | Focal cross-entropy with class weights {yes: 1.0, no: 2.5} |
| Decoding | Greedy first-token + logit-margin abstention |

### Retrieval

| Component | Value |
|-----------|-------|
| Embedder | pritamdeka/S-PubMedBert-MS-MARCO (768-dim) |
| Index | FAISS Inner-Product over L2-normalised embeddings |
| Chunking | 300-word windows with 50-word overlap |
| Default top-K | 5 passages |

### Confidence-margin abstention

CogniMed reads the model's softmax probability for the yes and no tokens at the first generation step. After normalising over those two labels, predictions where P(yes) falls within [0.5 − margin, 0.5 + margin] (default margin = 0.15) are returned as UNCERTAIN rather than committing to a low-confidence label. This trades coverage for calibration.

## ⚠️ Limitations

- **In-distribution only.** CogniMed is calibrated for the PubMedQA evidence-question distribution. Out-of-distribution queries are not reliably handled.
- **No causal reasoning.** Verdicts reflect statistical patterns in retrieved literature, not causal medical reasoning.
- **Publication bias.** The retrieval corpus inherits PubMed's publication bias — positive findings are over-represented.
- **Not for clinical use.** This is a research and educational tool. Always consult a qualified healthcare professional for medical decisions.

## 📄 License

MIT — see LICENSE for details.

## 👤 Author

**Hrishita Panjetha**

Master's Dissertation Project · 2026

---

<div align="center">

⭐ If you find CogniMed useful, consider giving the [HF Space](https://huggingface.co/spaces/Hrishita-P/cognimed) a like!

</div>
COGNIMED_README_END