"""اختبارات مُطبِّع النصّ العربي (المبني على CAMeL Tools).

يوثّق ويحمي السلوك الشكلي للتطبيع، ويثبّت أنّ التطبيع اللغوي الأساسي يعمل
عبر CAMeL دون تغيير عقد المخرجات الذي تعتمد عليه الطبقات الحتمية.
"""

from app.domain.text_preprocessing import normalize_arabic
# نفس الدالة يجب أن تكون مُعاد تصديرها من red_flag_engine (توافق خلفي).
from app.domain.red_flag_engine import normalize_arabic as normalize_via_red_flag


def test_empty_and_whitespace():
    assert normalize_arabic("") == ""
    assert normalize_arabic("   ") == ""


def test_alef_unification():
    # أ إ آ ٱ → ا (عبر CAMeL normalize_alef_ar)
    assert normalize_arabic("أحمد") == "احمد"
    assert normalize_arabic("إلتهاب") == "التهاب"
    assert normalize_arabic("آلام") == "الام"


def test_yaa_and_hamza_and_teh_marbuta():
    assert normalize_arabic("مستشفى") == "مستشفي"      # ى → ي (CAMeL)
    assert normalize_arabic("رئة") == "ريه"            # ئ → ي (إكمال) + ة → ه (CAMeL)
    assert normalize_arabic("لؤلؤ") == "لولو"          # ؤ → و (إكمال)
    assert normalize_arabic("حرارة") == "حراره"        # ة → ه (CAMeL)


def test_diacritics_and_tatweel_removed():
    assert normalize_arabic("صَدْر") == "صدر"
    assert normalize_arabic("الصُّداع") == "الصداع"
    assert normalize_arabic("مـــريض") == "مريض"       # تطويل


def test_arabic_indic_digits_converted():
    assert normalize_arabic("عمره ٤٠ يوم") == "عمره 40 يوم"
    assert normalize_arabic("٣٨ درجة") == "38 درجه"


def test_latin_tokens_preserved_as_latin_but_lowercased():
    # الأحرف اللاتينية لا تُشوَّه؛ تُصغَّر والشرطة تُعامَل كترقيم.
    assert normalize_arabic("COVID-19") == "covid 19"
    out = normalize_arabic("عندي حرارة COVID و ضيق تنفس")
    assert "covid" in out and "حراره" in out


def test_punctuation_and_whitespace_normalized():
    assert normalize_arabic("ألم،  صدر!") == "الم صدر"
    assert normalize_arabic("  ألم   صدر  ") == "الم صدر"


def test_formal_and_dialect_forms_collapse_to_same():
    # الغرض الأساس: صيغتان شكليتان لنفس الكلمة تتطابقان بعد التطبيع.
    assert normalize_arabic("ألم صَدر") == normalize_arabic("الم صدر")


def test_reexport_from_red_flag_engine_is_same_function():
    # التوافق الخلفي: الواردات القديمة ما زالت تعمل وتعطي نفس النتيجة.
    assert normalize_via_red_flag is normalize_arabic
    assert normalize_via_red_flag("أإآ ة ٥") == "ااا ه 5"
