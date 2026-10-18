"""Numeric CoT-shape features.

The classifier receives only the fixed numeric FEATURE_NAMES below. Raw CoT text
is used inside this extractor to compute counts/embeddings/NLL, but no token or
string is passed on to the downstream classifier.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import ruptures as rpt
import torch
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

FEATURE_NAMES = (
    "hedge_rate",
    "backtrack_rate",
    "phase_change_rate",
    "last_phase_share",
    "question_rate",
    "enumeration_rate",
    "code_fence_rate",
    "mean_nll",
    "nll_slope",
)
LENGTH_FEATURE_NAMES = ("n_chars", "n_sents")
ALL_FEATURE_NAMES = FEATURE_NAMES + LENGTH_FEATURE_NAMES

# Closed, versioned lexicon. Changing it changes the feature definition.
HEDGES = (
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "likely",
    "unlikely",
    "seems",
    "appears",
    "might",
    "could",
    "should",
    "roughly",
    "approximately",
    "unclear",
    "not sure",
    "i think",
    "i suspect",
    "it may",
    "it might",
    "presumably",
    "arguably",
    "tentatively",
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_ENUM = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _phrase_count(text: str, phrase: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.I))


def basic_features(text: str) -> dict[str, float]:
    sents = sentences(text)
    n_sents = max(1, len(sents))
    low = text.lower()
    return {
        "hedge_rate": float(sum(_phrase_count(low, h) for h in HEDGES) / n_sents),
        "question_rate": float(sum("?" in s for s in sents) / n_sents),
        "enumeration_rate": float(len(_ENUM.findall(text)) / n_sents),
        "code_fence_rate": float(text.count("```") / n_sents),
        "n_chars": float(len(text)),
        "n_sents": float(len(sents)),
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else 0.0


def backtrack_rate(emb: np.ndarray) -> float:
    """Repeated 2-sentence window followed by a divergent continuation.

    Event j is counted when a prior non-overlapping 2-sentence window is very
    similar (cos >= .85) but the next sentence diverges (cos < .5).
    """
    n = len(emb)
    if n < 5:
        return 0.0
    wins = np.asarray([(emb[i] + emb[i + 1]) / 2 for i in range(n - 1)])
    events: set[int] = set()
    for j in range(2, n - 2):
        for i in range(0, j - 1):
            if i + 2 >= n or j + 2 >= n:
                continue
            if _cos(wins[i], wins[j]) >= 0.85 and _cos(emb[i + 2], emb[j + 2]) < 0.5:
                events.add(j)
                break
    return float(len(events) / n)


def phase_features(emb: np.ndarray) -> tuple[float, float]:
    n = len(emb)
    if n < 3:
        return 0.0, 1.0
    try:
        breaks = rpt.Pelt(model="rbf").fit(emb).predict(pen=10)
    except Exception:
        return 0.0, 1.0
    internal = [int(x) for x in breaks if int(x) < n]
    last_start = internal[-1] if internal else 0
    return float(len(internal) / n), float((n - last_start) / n)


class ShapeExtractor:
    def __init__(
        self,
        models_path: str | Path = "config/models.yaml",
        *,
        device: str = "cpu",
        allow_unpinned_ref_lm: bool = False,
        nll_chunk_tokens: int = 256,
    ) -> None:
        cfg = yaml.safe_load(Path(models_path).read_text(encoding="utf-8"))["local"]
        self.embedder_id = cfg["embedder"]
        self.ref_lm_id = cfg["ref_lm"]["id"]
        self.ref_lm_revision = cfg["ref_lm"].get("revision")
        if (
            isinstance(self.ref_lm_revision, str)
            and self.ref_lm_revision.startswith("[")
            and not allow_unpinned_ref_lm
        ):
            raise RuntimeError(
                "config/models.yaml still has a placeholder ref_lm revision; pin the HF commit "
                "or pass --allow-unpinned-ref-lm for a non-reproducible smoke run"
            )
        if isinstance(self.ref_lm_revision, str) and self.ref_lm_revision.startswith(
            "["
        ):
            self.ref_lm_revision = None
        self.device = device
        self.nll_chunk_tokens = nll_chunk_tokens
        self._embedder: SentenceTransformer | None = None
        self._tok = None
        self._lm = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedder_id, device=self.device)
        return self._embedder

    def _load_lm(self):
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(
                self.ref_lm_id, revision=self.ref_lm_revision
            )
        if self._lm is None:
            self._lm = (
                AutoModelForCausalLM.from_pretrained(
                    self.ref_lm_id,
                    revision=self.ref_lm_revision,
                    torch_dtype=torch.float32,
                )
                .to(self.device)
                .eval()
            )
        return self._tok, self._lm

    def nll_features(self, text: str) -> tuple[float, float]:
        """Token NLL mean and position slope with rolling KV-cache context.

        Chunking controls only how many new tokens are evaluated per forward pass;
        the model keeps the full autoregressive past, so chunk boundaries do not
        reset context.
        """
        tok, lm = self._load_lm()
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if len(ids) < 2:
            return 0.0, 0.0
        losses: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        n = len(ids)
        past = None
        pos = 0
        with torch.inference_mode():
            while pos < n - 1:
                input_end = min(n - 1, pos + self.nll_chunk_tokens)
                chunk = ids[pos:input_end].unsqueeze(0).to(self.device)
                out = lm(input_ids=chunk, past_key_values=past, use_cache=True)
                logits = out.logits[0]
                target = ids[pos + 1 : input_end + 1].to(self.device)
                per = (
                    F.cross_entropy(logits, target, reduction="none")
                    .detach()
                    .cpu()
                    .numpy()
                )
                losses.append(per)
                positions.append(np.arange(pos + 1, input_end + 1, dtype=float))
                past = out.past_key_values
                pos = input_end
                del logits, target, chunk, out
        y = np.concatenate(losses)
        x = np.concatenate(positions)
        mean = float(np.mean(y))
        if len(y) < 2 or float(x.max()) == float(x.min()):
            slope = 0.0
        else:
            xn = (x - x.min()) / (x.max() - x.min())
            slope = float(np.polyfit(xn, y, 1)[0])
        return mean, slope

    def extract(
        self, text: str, *, model_features: bool = True, compute_nll: bool = True
    ) -> dict[str, float]:
        out = basic_features(text)
        sents = sentences(text)
        if model_features and sents:
            emb = self.embedder.encode(
                sents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out["backtrack_rate"] = backtrack_rate(emb)
            phase_rate, last_share = phase_features(emb)
            out["phase_change_rate"] = phase_rate
            out["last_phase_share"] = last_share
        else:
            out["backtrack_rate"] = 0.0
            out["phase_change_rate"] = 0.0
            out["last_phase_share"] = 1.0

        if model_features and compute_nll:
            out["mean_nll"], out["nll_slope"] = self.nll_features(text)
        else:
            out["mean_nll"], out["nll_slope"] = 0.0, 0.0
        return {name: float(out[name]) for name in ALL_FEATURE_NAMES}
