"""Universal Category System (UCS): carregar a lista oficial e utilitários de nomes."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
UCS_CSV = DATA_DIR / "ucs.csv"


@dataclass(frozen=True)
class UCSCategory:
    category: str
    subcategory: str
    catid: str
    catshort: str
    explanation: str
    synonyms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def full(self) -> str:
        return f"{self.category}-{self.subcategory}"

    def prompts(self) -> list[str]:
        """Frases usadas para descrever a categoria ao modelo CLAP."""
        cat = self.category.lower()
        sub = self.subcategory.lower()
        out = []
        if sub in ("misc", "general"):
            out.append(f"the sound of {cat}")
        else:
            out.append(f"the sound of {sub} {cat}")
        if self.explanation:
            out.append(self.explanation)
        if self.synonyms:
            out.append("the sound of " + ", ".join(self.synonyms[:6]))
        return out

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "catid": self.catid,
            "catshort": self.catshort,
            "explanation": self.explanation,
            "synonyms": list(self.synonyms),
        }


@lru_cache(maxsize=1)
def load_ucs() -> tuple[UCSCategory, ...]:
    cats = []
    with open(UCS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 4 or not row[2].strip():
                continue
            if row[0].strip().lower() == "category":  # cabeçalho, se existir
                continue
            synonyms = tuple(s.strip() for s in (row[5] if len(row) > 5 else "").split(",") if s.strip())
            cats.append(
                UCSCategory(
                    category=row[0].strip(),
                    subcategory=row[1].strip(),
                    catid=row[2].strip(),
                    catshort=row[3].strip(),
                    explanation=(row[4] if len(row) > 4 else "").strip(),
                    synonyms=synonyms,
                )
            )
    return tuple(cats)


@lru_cache(maxsize=1)
def by_catid() -> dict[str, UCSCategory]:
    return {c.catid: c for c in load_ucs()}


def lookup(catid: str) -> UCSCategory | None:
    return by_catid().get(catid)


# --- Nomes de ficheiro -----------------------------------------------------

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _clean_part(s: str) -> str:
    s = _INVALID.sub("", s or "")
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def build_filename(catid: str, fxname: str, creator_id: str, source_id: str,
                   user_data: str = "", ext: str = ".wav") -> str:
    """Nome UCS: CatID_FXName_CreatorID_SourceID[_UserData].wav"""
    parts = [_clean_part(catid) or "UNKNOWN", _clean_part(fxname) or "Untitled",
             _clean_part(creator_id) or "XX", _clean_part(source_id) or "XX"]
    if user_data.strip():
        parts.append(_clean_part(user_data))
    return "_".join(parts) + ext


def parse_filename(stem: str) -> dict:
    """Se o nome já seguir o UCS, extrai os campos."""
    parts = stem.split("_")
    if len(parts) >= 2 and lookup(parts[0]):
        out = {"catid": parts[0], "fxname": parts[1]}
        if len(parts) >= 3:
            out["creator_id"] = parts[2]
        if len(parts) >= 4:
            out["source_id"] = parts[3]
        return out
    return {}
