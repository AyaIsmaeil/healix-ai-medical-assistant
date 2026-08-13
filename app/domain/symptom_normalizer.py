"""
Healix — Symptom Normalizer
تطبيع أسماء الأعراض العربية إلى صيغ معيارية قبل التخزين والـ ML.

يعتمد على ``symptom_evidence_map.yaml`` (ربط DDXPlus E_* codes) — لا قائمة
مُختلَقة. يُسقِط الشظايا اللغوية (يزداد، اليمين، ثقيل...) التي ليست أعراضاً
بل وصف OLDCARTS.

مراجع: DDXPlus release_evidences.json — نفس مصدر تدريب النموذج.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import yaml

from app.domain.text_preprocessing import normalize_arabic

_DICTIONARIES_DIR = Path(__file__).resolve().parent.parent / "dictionaries"
_SYMPTOM_MAP_PATH = _DICTIONARIES_DIR / "symptom_evidence_map.yaml"

# كلمات وصفية من OLDCARTS — ليست أعراضاً مستقلة (مأخوذة من أنماط المقابلة)
_DESCRIPTOR_STOPWORDS: FrozenSet[str] = frozenset({
    "يزداد", "يزداد الضيق", "يزداد الضغط", "يزداد عند الراحة",
    "شديد", "شديدة", "خفيف", "متوسط", "ثقيل", "حاد", "نابض", "حارق",
    "اليمين", "اليسار", "يمين", "يسار", "اعلى", "اسفل", "أعلى", "أسفل",
    "عند الراحة", "عند المجهود", "مدة طويلة", "منذ", "يوم", "يومين",
    "نعم", "لا", "معا", "مع", "ابدا", "أبدا",
})

# توسيعات إضافية موثّقة — نفس مفهوم chest_pain بـ symptom_evidence_map
_EXTRA_CANONICAL: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ألم صدر", ("ضغط على الصدر", "ضغط صدر", "ضغط بالصدر", "ضغط في الصدر")),
)


@lru_cache(maxsize=1)
def _load_alias_index() -> List[Tuple[str, str]]:
    """(alias_normalized, canonical_arabic) مرتّبة من الأطول للأقصر."""
    aliases: List[Tuple[str, str]] = []

    if _SYMPTOM_MAP_PATH.exists():
        with _SYMPTOM_MAP_PATH.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for entry in data.get("mappings", []):
            canonical = entry["names"][0]
            for name in entry.get("names", []):
                aliases.append((normalize_arabic(name), canonical))

    for canonical, names in _EXTRA_CANONICAL:
        for name in names:
            aliases.append((normalize_arabic(name), canonical))

    aliases.sort(key=lambda pair: len(pair[0]), reverse=True)
    return aliases


def is_descriptor_fragment(text: str) -> bool:
    """هل النصّ شظية وصفية (OLDCARTS) وليس عرضاً؟"""
    normalized = normalize_arabic(text.strip())
    if not normalized:
        return True
    if normalized in _DESCRIPTOR_STOPWORDS:
        return True
    if len(normalized) < 4 and normalized not in {"حمى", "كحة"}:
        return True
    return False


def normalize_symptom_text(text: str) -> Optional[str]:
    """يُعيد الاسم المعياري للعرض، أو ``None`` إن كان وصفاً لا عرضاً."""
    raw = text.strip()
    if not raw or is_descriptor_fragment(raw):
        return None

    normalized = normalize_arabic(raw)
    for alias, canonical in _load_alias_index():
        if normalized == alias:
            return canonical

    for alias, canonical in _load_alias_index():
        if len(normalized) >= max(4, len(alias) // 2) and alias in normalized:
            return canonical

    # عرض حرّ مُثبت — يُقبل إن كان طويلاً بما يكفي
    if len(normalized) >= 4:
        return raw
    return None


def normalize_symptoms_for_storage(
    symptoms: List,
) -> List:
    """يُطبّع قائمة ``Symptom`` — يدمج المكرّر ويُسقِط الشظايا."""
    from app.domain.conversation import Symptom

    seen: Dict[tuple, Symptom] = {}
    for symptom in symptoms:
        if symptom.negated:
            canonical = symptom.text.strip()
            key = (canonical, True)
        else:
            canonical = normalize_symptom_text(symptom.text)
            if canonical is None:
                continue
            key = (canonical, False)

        existing = seen.get(key)
        if existing is None or (symptom.is_patient_stated and not existing.is_patient_stated):
            seen[key] = Symptom(
                text=canonical,
                negated=symptom.negated,
                confidence=symptom.confidence,
                evidence=symptom.evidence,
                source=symptom.source,
                turn_number=symptom.turn_number,
            )
    return list(seen.values())
