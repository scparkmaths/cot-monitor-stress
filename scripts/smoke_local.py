"""Smoke test: the two local models used for content-blind features load and run on CPU."""

from pathlib import Path

import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

config_path = Path(__file__).resolve().parents[1] / "config/models.yaml"
with config_path.open(encoding="utf-8") as f:
    local = yaml.safe_load(f)["local"]
EMBEDDER = local["embedder"]
REF_LM = local["ref_lm"]["id"]
REF_LM_REVISION = local["ref_lm"]["revision"]

emb = SentenceTransformer(EMBEDDER, device="cpu")
vecs = emb.encode(["hello", "world"])
print("embedder output shape:", vecs.shape)  # expect (2, 384)

tok = AutoTokenizer.from_pretrained(REF_LM, revision=REF_LM_REVISION)
model = AutoModelForCausalLM.from_pretrained(
    REF_LM, revision=REF_LM_REVISION, torch_dtype=torch.float32
)
model.eval()

ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
with torch.no_grad():
    out = model(**ids, labels=ids["input_ids"])
print(
    "ref LM mean token NLL:", round(float(out.loss), 3)
)  # expect a float, roughly 1-5
