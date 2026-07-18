"""
اختبارات وحدة لخدمة استخراج الأعراض (SymptomExtractor).

تُستخدم كائنات وهمية (Fakes) بدل النموذج الحقيقي، فلا حاجة لتحميل الأوزان،
ما يجعل المنطق قابلاً للاختبار بشكل معزول وسريع.
"""

import torch

from app.services.symptom_extractor import ExtractedSymptom, SymptomExtractor

# خريطة التصنيفات المطابقة لنموذج NER
ID2LABEL = {
    0: "O",
    1: "B-SYMPTOM_POS",
    2: "I-SYMPTOM_POS",
    3: "B-SYMPTOM_NEG",
    4: "I-SYMPTOM_NEG",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)


class _FakeEncoding:
    """يحاكي مخرجات الـ Tokenizer السريع (BatchEncoding)."""

    def __init__(self, seq_len, word_ids):
        self._data = {"input_ids": torch.zeros((1, seq_len), dtype=torch.long)}
        self._word_ids = word_ids

    def items(self):
        return self._data.items()

    def word_ids(self, batch_index=0):
        return self._word_ids


class _FakeTokenizer:
    is_fast = True

    def __init__(self, word_ids):
        self._word_ids = word_ids

    def __call__(self, words, **kwargs):
        return _FakeEncoding(len(self._word_ids), self._word_ids)


class _FakeModel:
    """نموذج وهمي يُرجع logits ثابتة لكل رمز حسب التصنيف المطلوب."""

    def __init__(self, token_label_ids):
        self._token_label_ids = token_label_ids

    def __call__(self, **inputs):
        logits = torch.full((1, len(self._token_label_ids), NUM_LABELS), -5.0)
        for idx, label_id in enumerate(self._token_label_ids):
            logits[0, idx, label_id] = 5.0  # ثقة عالية بعد softmax

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


class _FakeLoader:
    def __init__(self, word_ids, token_label_ids):
        self.tokenizer = _FakeTokenizer(word_ids)
        self.model = _FakeModel(token_label_ids)
        self.id2label = ID2LABEL
        self.device = torch.device("cpu")


def _extractor(word_ids, token_labels, threshold=0.5):
    token_label_ids = [LABEL2ID[l] for l in token_labels]
    loader = _FakeLoader(word_ids, token_label_ids)
    return SymptomExtractor(loader, confidence_threshold=threshold, max_length=128)


# ----------------------------------------------------------------------
# اختبارات المنطق الخالص
# ----------------------------------------------------------------------
def test_parse_label():
    assert SymptomExtractor._parse_label("O") == (None, None)
    assert SymptomExtractor._parse_label("B-SYMPTOM_POS") == ("B", "SYMPTOM_POS")
    assert SymptomExtractor._parse_label("I-SYMPTOM_NEG") == ("I", "SYMPTOM_NEG")


def test_empty_and_whitespace_text_returns_empty():
    ext = _extractor([None], ["O"])
    assert ext.extract("") == []
    assert ext.extract("   ") == []


def test_merges_positive_and_negative_symptoms_in_order():
    # الكلمات: أعاني من صداع شديد ولا يوجد كحة  (7 كلمات)
    # "صداع" مكوّنة من رمزين فرعيين لاختبار منطق أول رمز فرعي.
    text = "أعاني من صداع شديد ولا يوجد كحة"
    word_ids = [None, 0, 1, 2, 2, 3, 4, 5, 6, None]
    token_labels = [
        "O",              # CLS
        "O",              # أعاني
        "O",              # من
        "B-SYMPTOM_POS",  # صداع (رمز أول)
        "I-SYMPTOM_POS",  # صداع (رمز تابع — يُتجاهل)
        "I-SYMPTOM_POS",  # شديد
        "O",              # ولا
        "O",              # يوجد
        "B-SYMPTOM_NEG",  # كحة
        "O",              # SEP
    ]
    result = _extractor(word_ids, token_labels).extract(text)

    assert result == [
        ExtractedSymptom(text="صداع شديد", negated=False, confidence=result[0].confidence),
        ExtractedSymptom(text="كحة", negated=True, confidence=result[1].confidence),
    ]
    assert result[0].negated is False
    assert result[1].negated is True
    assert all(0.0 <= s.confidence <= 1.0 for s in result)


def test_confidence_threshold_filters_low_confidence():
    # نفس التنبؤ لكن بعتبة عالية جداً — يجب ألا يُقبل أي عرض.
    text = "صداع"
    word_ids = [None, 0, None]
    token_labels = ["O", "B-SYMPTOM_POS", "O"]
    result = _extractor(word_ids, token_labels, threshold=0.99999).extract(text)
    assert result == []


def test_unique_symptoms_preserve_order():
    # "صداع" يتكرّر — يجب أن يظهر مرّة واحدة فقط بترتيب أوّل ظهور.
    text = "صداع وحمى وصداع"
    word_ids = [None, 0, 1, 2, None]
    token_labels = [
        "O",
        "B-SYMPTOM_POS",  # صداع
        "B-SYMPTOM_POS",  # وحمى
        "B-SYMPTOM_POS",  # وصداع -> نفس النص "صداع"؟ لا، النص مختلف
        "O",
    ]
    # ملاحظة: الكلمات مختلفة نصياً، لذا نتوقّع ثلاثة عناصر فريدة.
    result = _extractor(word_ids, token_labels).extract(text)
    assert [s.text for s in result] == ["صداع", "وحمى", "وصداع"]


def test_true_duplicate_is_deduplicated():
    text = "صداع صداع"
    word_ids = [None, 0, 1, None]
    token_labels = ["O", "B-SYMPTOM_POS", "B-SYMPTOM_POS", "O"]
    result = _extractor(word_ids, token_labels).extract(text)
    assert [s.text for s in result] == ["صداع"]
