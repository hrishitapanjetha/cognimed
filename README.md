---
title: CogniMed
emoji: 🧠
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.13.0
app_file: app.py
pinned: false
---

# CogniMed

AI cognition applied to medicine.

CogniMed is a Retrieval-Augmented Generation system for evidence-grounded
biomedical question answering. It pairs a domain-tuned sentence-transformer
for retrieval with a LoRA-fine-tuned BioGPT-Large language model for binary
yes / no clinical reasoning, grounded in PubMed-derived passages.

## Live Demo

https://huggingface.co/spaces/Hrishita-P/cognimed

## Architecture

Question -> S-PubMedBERT encoder -> FAISS retrieval over PubMedQA -> BioGPT-Large + LoRA reasoner -> Yes / No / Uncertain verdict with supporting passages.

## Stack

- Retriever: pritamdeka/S-PubMedBert-MS-MARCO + FAISS Inner-Product index
- Reasoner: microsoft/BioGPT-Large-PubMedQA + LoRA (rank 32, alpha 64)
- UI: Gradio with a beige palette
- Trained on the binary yes / no subset of PubMedQA

## Run Locally

    git clone https://github.com/hrishitapanjetha/cognimed.git
    cd cognimed
    pip install -r requirements.txt
    python app.py

## Limitations

CogniMed is calibrated for the PubMedQA evidence-question distribution. Out-of-distribution queries are not reliably handled. Not for clinical use.

## License

MIT

## Author

Hrishita Panjetha