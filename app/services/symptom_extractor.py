"""
Healix - Symptom Extractor
استخراج الأعراض من نص عربي باستخدام نموذج MARBERT (Token Classification / NER).

المنطق:
1. تقطيع النص إلى كلمات ثمّ ترميزه بالـ Tokenizer الخاص بـ MARBERT.
2. الاستدلال داخل ``torch.no_grad()``.
3. تحويل تنبؤات الرموز (sub-tokens) إلى تصنيفات على مستوى الكلمة (أول رمز فرعي).
4. دمج ``B-SYMPTOM_POS`` + ``I-SYMPTOM_POS`` في عبارة عرض واحدة (إيجابي).
5. دمج ``B-SYMPTOM_NEG`` + ``I-SYMPTOM_NEG`` في عبارة عرض منفي.
6. تجاهل ``O``، إرجاع الأعراض الفريدة فقط مع الحفاظ على ترتيب ظهورها.

الخدمة لا تحتوي منطق تشخيص، ولا تعتمد على LLM أو قواعد يدوية للاستخراج.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from app.exceptions import InferenceError
from app.services.marbert_service import ModelLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedSymptom:
    """كائن مجال يمثّل عرضاً مستخرجاً (مستقل عن طبقة الـ API)."""

    text: str
    negated: bool
    confidence: float


class SymptomExtractor:
    """
    خدمة استخراج الأعراض. تعتمد على ``ModelLoader`` عبر الحقن (DI).

    Args:
        model_loader: مُحمّل النموذج (يجب أن يكون قد نُفّذ ``load()``).
        confidence_threshold: الحد الأدنى للثقة لقبول العرض (قابل للضبط).
        max_length: أقصى طول للتسلسل عند الترميز.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        confidence_threshold: float = 0.5,
        max_length: int = 256,
    ) -> None:
        self._loader = model_loader
        self._threshold = float(confidence_threshold)
        self._max_length = int(max_length)
        # الاستدلال قراءة فقط لكن نحمي التمرير الأمامي لضمان سلامة الخيوط.
        self._infer_lock = threading.Lock()

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    # الواجهة العامة
    # ------------------------------------------------------------------
    def extract(self, text: str) -> List[ExtractedSymptom]:
        """
        استخراج الأعراض الفريدة من النص مع الحفاظ على ترتيب الظهور.

        Returns:
            قائمة ``ExtractedSymptom`` (قد تكون فارغة).
        """
        if not text or not text.strip():
            return []

        words = text.split()
        if not words:
            return []

        try:
            word_labels, word_confs = self._predict_word_labels(words)
        except Exception as exc:  # noqa: BLE001
            logger.exception("❌ فشل الاستدلال على النص")
            raise InferenceError(f"فشل استخراج الأعراض: {exc}") from exc

        entities = self._merge_entities(words, word_labels, word_confs)
        return self._deduplicate(entities)

    # ------------------------------------------------------------------
    # الاستدلال على مستوى الكلمة
    # ------------------------------------------------------------------
    def _predict_word_labels(
        self, words: List[str]
    ) -> Tuple[Dict[int, str], Dict[int, float]]:
        """
        تنفيذ الاستدلال وإرجاع تصنيف وثقة كل كلمة (اعتماداً على أول رمز فرعي).
        """
        tokenizer = self._loader.tokenizer
        model = self._loader.model
        id2label = self._loader.id2label
        device = self._loader.device

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        word_ids = encoding.word_ids(batch_index=0)
        inputs = {k: v.to(device) for k, v in encoding.items()}

        with self._infer_lock:
            with torch.no_grad():
                logits = model(**inputs).logits  # [1, seq_len, num_labels]

        probs = torch.softmax(logits, dim=-1)[0]  # [seq_len, num_labels]
        confidences, pred_ids = probs.max(dim=-1)  # [seq_len], [seq_len]

        word_labels: Dict[int, str] = {}
        word_confs: Dict[int, float] = {}
        seen: set = set()
        for tok_idx, w_id in enumerate(word_ids):
            if w_id is None or w_id in seen:
                continue  # رموز خاصة أو رموز فرعية تابعة — نأخذ أول رمز فقط
            seen.add(w_id)
            word_labels[w_id] = id2label.get(int(pred_ids[tok_idx]), "O")
            word_confs[w_id] = float(confidences[tok_idx])

        return word_labels, word_confs

    # ------------------------------------------------------------------
    # دمج تصنيفات BIO إلى عبارات
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_label(label: str) -> Tuple[Optional[str], Optional[str]]:
        """تفكيك تصنيف BIO إلى (البادئة، نوع الكيان). يُرجِع (None, None) للـ O."""
        if not label or label == "O" or "-" not in label:
            return None, None
        prefix, entity_type = label.split("-", 1)
        return prefix, entity_type

    def _merge_entities(
        self,
        words: List[str],
        word_labels: Dict[int, str],
        word_confs: Dict[int, float],
    ) -> List[ExtractedSymptom]:
        """دمج الكلمات المتتالية وفق مخطط BIO إلى عبارات أعراض كاملة."""
        entities: List[ExtractedSymptom] = []
        cur_words: List[str] = []
        cur_confs: List[float] = []
        cur_type: Optional[str] = None

        def flush() -> None:
            nonlocal cur_words, cur_confs, cur_type
            if cur_type is not None and cur_words:
                confidence = sum(cur_confs) / len(cur_confs)
                if confidence >= self._threshold:
                    entities.append(
                        ExtractedSymptom(
                            text=" ".join(cur_words),
                            negated=cur_type.upper().endswith("NEG"),
                            confidence=round(confidence, 4),
                        )
                    )
            cur_words = []
            cur_confs = []
            cur_type = None

        for w_id in range(len(words)):
            label = word_labels.get(w_id, "O")
            prefix, entity_type = self._parse_label(label)

            if prefix is None:  # O — نتجاهله وننهي أي كيان جارٍ
                flush()
                continue

            # استكمال نفس الكيان بـ I-، وإلا نبدأ كياناً جديداً
            # (نعامل I- الشاردة بلا B- كبداية كيان لزيادة المتانة).
            if prefix == "I" and cur_type == entity_type:
                cur_words.append(words[w_id])
                cur_confs.append(word_confs.get(w_id, 0.0))
            else:
                flush()
                cur_type = entity_type
                cur_words = [words[w_id]]
                cur_confs = [word_confs.get(w_id, 0.0)]

        flush()
        return entities

    @staticmethod
    def _deduplicate(entities: List[ExtractedSymptom]) -> List[ExtractedSymptom]:
        """إرجاع الأعراض الفريدة فقط مع الحفاظ على ترتيب أوّل ظهور."""
        seen: set = set()
        unique: List[ExtractedSymptom] = []
        for ent in entities:
            key = (ent.text, ent.negated)
            if key in seen:
                continue
            seen.add(key)
            unique.append(ent)
        return unique
