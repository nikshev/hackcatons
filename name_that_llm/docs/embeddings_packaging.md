# Offline Kaggle Packaging for Embedding Models (Runbook)

Status: experimental path only. The frozen-MiniLM local experiment lost to
flat char TF-IDF decisively (0.1345 vs 0.3008 OOF), so do not package
embeddings unless a future experiment reverses that verdict.

## Why this exists

Kaggle code competitions run with internet disabled. `pip install
sentence-transformers` and Hugging Face downloads fail there. The Kaggle base
image ships `torch` and `transformers`, but not `sentence-transformers`, so the
offline Notebook must avoid that dependency entirely.

## Packaging steps (only if embeddings ever win locally)

1. Vendor weights: upload the contents of the local Git-ignored `models/`
   directory (e.g. `all-MiniLM-L6-v2/`) as a private Kaggle dataset, e.g.
   `username/ntllm-minilm`. Record its version and file checksums.
2. Attach to the submission Notebook: the competition data plus the weights
   dataset. Paths look like
   `/kaggle/input/<slug>/pytorch_model.bin`.
3. In the Notebook, load with `transformers` only (no `sentence-transformers`):

   ```python
   import torch
   from transformers import AutoModel, AutoTokenizer

   weights = "/kaggle/input/<slug>"  # attached dataset directory
   tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
   model = AutoModel.from_pretrained(weights, local_files_only=True)
   model.eval()

   def embed(texts):
       batch = tokenizer(list(texts), padding=True, truncation=True,
                         max_length=512, return_tensors="pt")
       with torch.no_grad():
           hidden = model(**batch).last_hidden_state
       mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
       summed = (hidden * mask).sum(1)
       counts = mask.sum(1).clamp(min=1e-9)
       vectors = summed / counts
       return torch.nn.functional.normalize(vectors, p=2, dim=1).numpy()
   ```

4. Confirm versions: log `torch.__version__` and `transformers.__version__`
   from the Kaggle image in the run summary; they must satisfy the vendored
   `config.json` architecture before trusting outputs.
5. Keep the runtime budget in mind: encode the hidden test set once, then
   predict. Measure against the nine-hour CPU/GPU limit with margin.

## Local layout

- `models/` — Git-ignored vendored weights (never committed).
- Experiment configs in `configs/experiments/` record the model directory and
  the weights checksum appears in every tuning summary (`embedding_model_sha256`).
