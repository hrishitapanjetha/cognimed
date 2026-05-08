"""
ui/lora_inference.py
BioGPT-Large + LoRA inference + FAISS retriever.

Place in ui/ next to app.py. No dependency on models.inference.
"""

import re
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT      = Path(__file__).parent.parent
_CHECKPOINT_DIR = _REPO_ROOT / "models" / "checkpoints" / "lora_binary_best"
_BASE_MODEL_ID  = "microsoft/biogpt-large"

_YES_TOKENS = {"yes", "1", "true"}
_NO_TOKENS  = {"no",  "0", "false"}


def _clean_output(text: str) -> str:
    text = re.sub(r"<\s*/\s*[A-Za-z_][A-Za-z0-9_]*\s*>", "", text)
    text = re.sub(r"<\s*[A-Za-z_][A-Za-z0-9_]*\s*>", "", text)
    text = re.sub(r"<[^>]{0,50}>", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


class LoRAInferenceEngine:
    """Loads microsoft/biogpt-large + lora_binary_best and answers yes/no medical questions."""

    CONFIDENCE_MARGIN = 0.15

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        device: str = "auto",
        max_new_tokens: int = 4,
    ):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else _CHECKPOINT_DIR
        self.device         = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.model          = None
        self.tokenizer      = None
        self._loaded        = False

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_dir}")
        if not (self.checkpoint_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"adapter_config.json missing in {self.checkpoint_dir}")

    def load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("Run: pip install transformers peft accelerate safetensors") from e

        logger.info(f"Loading base model : {_BASE_MODEL_ID}")
        logger.info(f"Loading LoRA adapter: {self.checkpoint_dir}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.checkpoint_dir), trust_remote_code=True
            )
            logger.info("Tokenizer loaded from checkpoint.")
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_ID, trust_remote_code=True)
            logger.info("Tokenizer loaded from HuggingFace hub.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        import torch
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        base = AutoModelForCausalLM.from_pretrained(_BASE_MODEL_ID, **load_kwargs)
        self.model = PeftModel.from_pretrained(base, str(self.checkpoint_dir))
        self.model.eval()

        if self.device != "cpu":
            self.model = self.model.to(self.device)
        logger.info(f"Model placed on device: {self.device} (dtype={dtype})")

        self._loaded = True
        logger.info("✅ BioGPT-Large + LoRA adapter loaded successfully.")

    def answer(self, question: str, context_chunks: Optional[list] = None) -> dict:
        if not self._loaded:
            self.load()

        import torch
        import torch.nn.functional as F
        prompt = self._build_prompt(question, context_chunks)
        t0 = time.time()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True
        )
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        output_ids = gen_out.sequences

        first_logits = gen_out.scores[0][0]
        probs = F.softmax(first_logits, dim=-1)

        def _first_id(word: str):
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            return ids[0] if ids else None

        yes_id = _first_id("yes")
        no_id  = _first_id("no")

        p_yes_raw = float(probs[yes_id].item()) if yes_id is not None else 0.0
        p_no_raw  = float(probs[no_id].item())  if no_id  is not None else 0.0
        total = p_yes_raw + p_no_raw
        if total > 0:
            p_yes = p_yes_raw / total
            p_no  = p_no_raw  / total
        else:
            p_yes = p_no = 0.5

        if abs(p_yes - 0.5) < self.CONFIDENCE_MARGIN:
            label = "UNCERTAIN"
        elif p_yes > p_no:
            label = "YES"
        else:
            label = "NO"

        latency_ms = (time.time() - t0) * 1000
        input_len  = inputs["input_ids"].shape[1]
        raw        = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        generated  = _clean_output(raw) or raw

        return {
            "answer":          generated,
            "predicted_label": label,
            "p_yes":           p_yes,
            "p_no":            p_no,
            "confidence":      max(p_yes, p_no),
            "latency_ms":      latency_ms,
            "prompt":          prompt,
            "safe":            True,
            "emergency":       False,
        }

    def _build_prompt(self, question: str, chunks: Optional[list]) -> str:
        if chunks:
            raw_ctx = " ".join(
                (c["chunk"] if isinstance(c, dict) else str(c)).strip()
                for c in chunks[:5]
            )
            ctx = re.sub(r"\s+", " ", raw_ctx).strip()
            if len(ctx) > 2000:
                ctx = ctx[:2000].rstrip()
            return (
                "You are a medical expert.\n\n"
                f"Context: {ctx}\n\n"
                f"Question: {question.strip()}\n\n"
                "Answer ONLY yes or no.\n"
                "Answer: "
            )
        return (
            "You are a medical expert.\n\n"
            f"Question: {question.strip()}\n\n"
            "Answer ONLY yes or no.\n"
            "Answer: "
        )

    def _extract_label(self, text: str) -> str:
        if not text.strip():
            return "UNCERTAIN"
        first = text.strip().lower().split()[0].rstrip(".,;:!?")
        if first in _YES_TOKENS: return "YES"
        if first in _NO_TOKENS:  return "NO"
        snippet = text[:30].lower()
        if "yes" in snippet: return "YES"
        if "no"  in snippet: return "NO"
        return "UNCERTAIN"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"


class FAISSRetriever:
    """Loads the FAISS index from data/index/faiss.index."""

    def __init__(self):
        self._retriever  = None
        self._manual     = False
        self._available  = False
        self._faiss_index = None
        self._chunks      = None
        self._metadata    = None
        self._embedder    = None

    def load(self):
        import sys
        sys.path.insert(0, str(_REPO_ROOT))

        try:
            from retrieval.retriever import Retriever
            import inspect

            sig    = inspect.signature(Retriever.__init__)
            params = list(sig.parameters.keys())
            index_dir = str(_REPO_ROOT / "data" / "index")

            if "index_path" in params:
                self._retriever = Retriever(index_path=str(_REPO_ROOT / "data" / "index" / "faiss.index"))
            elif "index_dir" in params:
                self._retriever = Retriever(index_dir=index_dir)
            elif "data_dir" in params:
                self._retriever = Retriever(data_dir=index_dir)
            else:
                self._retriever = Retriever()

            for meth in ("load", "initialize", "load_index", "build"):
                if hasattr(self._retriever, meth):
                    try:
                        getattr(self._retriever, meth)()
                        break
                    except Exception:
                        pass

            self._available = True
            logger.info("✅ FAISS retriever loaded via retrieval.retriever.Retriever")
            return

        except Exception as e:
            logger.info(f"retrieval.retriever not usable ({e}), trying manual FAISS load.")

        try:
            import faiss
            import numpy as np
            import json

            index_path  = _REPO_ROOT / "data" / "index" / "faiss.index"
            chunks_path = _REPO_ROOT / "data" / "index" / "corpus_chunks.json"
            meta_path   = _REPO_ROOT / "data" / "index" / "corpus_metadata.json"

            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

            self._faiss_index = faiss.read_index(str(index_path))
            with open(chunks_path)  as f: self._chunks   = json.load(f)
            with open(meta_path)    as f: self._metadata = json.load(f)

            from embeddings.pubmedbert_embedder import PubMedBERTEmbedder
            self._embedder = PubMedBERTEmbedder()

            for meth in ("load", "initialize", "setup", "load_model", "build", "fit"):
                if hasattr(self._embedder, meth):
                    try:
                        getattr(self._embedder, meth)()
                        logger.info(f"PubMedBERTEmbedder initialised via .{meth}()")
                        break
                    except Exception as em:
                        logger.debug(f"  .{meth}() failed: {em}")
            else:
                logger.info("PubMedBERTEmbedder has no explicit load method — assuming auto-init.")

            self._manual    = True
            self._available = True
            logger.info("✅ FAISS retriever loaded manually (faiss + PubMedBERTEmbedder).")

        except Exception as e:
            self._available = False
            raise RuntimeError(f"FAISS retriever unavailable: {e}") from e

    def retrieve(self, query: str, top_k: int = 5) -> list:
        if not self._available:
            return []

        if self._retriever is not None:
            try:
                for meth in ("retrieve", "search", "query", "get_relevant_chunks"):
                    if hasattr(self._retriever, meth):
                        raw = getattr(self._retriever, meth)(query, top_k=top_k)
                        return self._normalise(raw)
            except Exception as e:
                logger.warning(f"Retriever search failed: {e}")
                return []

        if self._manual:
            try:
                import numpy as np
                import faiss as faiss_lib

                q_vec = None
                for enc in ("encode", "embed", "get_embeddings", "transform", "encode_queries"):
                    if hasattr(self._embedder, enc):
                        try:
                            result = getattr(self._embedder, enc)([query])
                            q_vec  = result
                            break
                        except Exception:
                            try:
                                result = getattr(self._embedder, enc)(query)
                                q_vec  = result
                                break
                            except Exception:
                                pass

                if q_vec is None:
                    raise RuntimeError("Could not encode query — no working encode method found")

                q_vec = np.array(q_vec, dtype="float32")
                if q_vec.ndim == 1:
                    q_vec = q_vec.reshape(1, -1)

                faiss_lib.normalize_L2(q_vec)
                scores, indices = self._faiss_index.search(q_vec, top_k)

                results = []
                for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                    if idx < 0:
                        continue
                    chunk = (
                        self._chunks[idx]
                        if isinstance(self._chunks, list)
                        else self._chunks.get(str(idx), "")
                    )
                    meta = (
                        self._metadata[idx]
                        if isinstance(self._metadata, list)
                        else self._metadata.get(str(idx), {})
                    )
                    results.append({
                        "rank":     rank + 1,
                        "chunk":    chunk,
                        "score":    float(score),
                        "metadata": meta,
                    })
                return results

            except Exception as e:
                logger.warning(f"Manual FAISS search failed: {e}")
                return []

        return []

    @staticmethod
    def _normalise(raw) -> list:
        if not raw:
            return []
        out = []
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                item.setdefault("rank",     i + 1)
                item.setdefault("chunk",    item.get("text", item.get("content", str(item))))
                item.setdefault("score",    item.get("similarity", 0.0))
                item.setdefault("metadata", item.get("meta", {}))
                out.append(item)
            else:
                out.append({"rank": i+1, "chunk": str(item), "score": 0.0, "metadata": {}})
        return out