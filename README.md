# GEO Metadata Cleaner — Stage 0

Stage 0 is a "clean slate" scaffold that is:
- Single-PC friendly (Ubuntu + NVIDIA GPU supported)
- Deterministic canonicalization (single text truth for offsets/negation)
- Multi-source proposal (regex + lexicon + optional neural extractor)
- Hybrid mapping (exact + lexical BM25 + vector) with deterministic fusion
- Validation hooks (negation + optional NLI gating + soft coherence signals)
- Built-in artifact outputs (JSON/CSV) suitable for paper experiments later

## Setup (Conda)
```bash
conda env create -f environment.yaml
conda activate geo-cleaner
pip install -e .
