from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contracts import Mention


def _sorted_mentions(ms: List[Mention]) -> List[Mention]:
    # deterministic ordering
    return sorted(
        ms, key=lambda m: (m.source_field, m.start, m.end, m.label, m.surface_form)
    )


@dataclass(frozen=True)
class ExtractorConfig:
    labels: Tuple[str, ...]
    device: str = "cpu"
    threshold: float = 0.0  # keep recall high by default
    seed: int = 0  # for deterministic wrappers that support it


class BaseExtractor:
    def extract(self, raw_fields: Mapping[str, str]) -> List[Mention]:
        raise NotImplementedError


class RegexExtractor(BaseExtractor):
    """
    Deterministic, dependency-free extractor useful for tests and smoke checks.
    Provide patterns keyed by label.
    """

    def __init__(self, patterns: Dict[str, str], conf: float = 1.0):
        import re

        self._re = re
        self._compiled = {
            lbl: re.compile(pat, flags=re.IGNORECASE) for lbl, pat in patterns.items()
        }
        self._conf = conf

    def extract(self, raw_fields: Mapping[str, str]) -> List[Mention]:
        out: List[Mention] = []
        for field_key, text in raw_fields.items():
            if not isinstance(text, str) or not text:
                continue
            for lbl, rx in self._compiled.items():
                for m in rx.finditer(text):
                    if m.start() == m.end():
                        continue
                    out.append(
                        Mention(
                            label=lbl,
                            surface_form=text[m.start() : m.end()],
                            source_field=field_key,
                            start=m.start(),
                            end=m.end(),
                            extractor_conf=float(self._conf),
                        )
                    )
        return _sorted_mentions(out)


class GLiNERExtractor(BaseExtractor):
    """
    High-recall extractor wrapper operating per RAW FIELD.
    Offsets are returned in field coordinates (field-scoped).
    """

    def __init__(self, model_name_or_path: str, cfg: ExtractorConfig):
        self.cfg = cfg
        self.model_name_or_path = model_name_or_path
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        # Seeds for determinism (best-effort). CPU inference is typically more stable than GPU.
        try:
            import random
            import numpy as np
            import torch

            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.seed)
        except Exception:
            pass

        from gliner import GLiNER  # type: ignore

        self._model = GLiNER.from_pretrained(self.model_name_or_path)

        # Some GLiNER versions support setting device; if not, it will run on default.
        try:
            import torch

            device = self.cfg.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._model.to(device)
        except Exception:
            pass

    def extract(self, raw_fields: Mapping[str, str]) -> List[Mention]:
        self._ensure_loaded()
        assert self._model is not None

        out: List[Mention] = []
        labels = list(self.cfg.labels)

        for field_key, text in raw_fields.items():
            if not isinstance(text, str) or not text.strip():
                continue

            # GLiNER API varies slightly across versions; handle both shapes.
            try:
                ents = self._model.predict_entities(
                    text, labels, threshold=self.cfg.threshold
                )
            except TypeError:
                ents = self._model.predict_entities(text, labels)

            for e in ents or []:
                try:
                    start = int(e["start"])
                    end = int(e["end"])
                    lbl = str(e["label"])
                    score = float(e.get("score", 0.0))
                except Exception:
                    continue

                if (
                    start < 0
                    or end < 0
                    or end < start
                    or end > len(text)
                    or start == end
                ):
                    continue
                if score < self.cfg.threshold:
                    continue

                surface = text[start:end]
                out.append(
                    Mention(
                        label=lbl,
                        surface_form=surface,
                        source_field=field_key,
                        start=start,
                        end=end,
                        extractor_conf=max(0.0, min(1.0, score)),
                    )
                )

        return _sorted_mentions(out)
