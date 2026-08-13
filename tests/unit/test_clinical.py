"""اختبارات وحدة للمعرفة السريرية والمُخطِّط (domain.clinical)."""

from collections import namedtuple

from app.domain import clinical
from app.domain.red_flag_engine import RedFlagEngine
from app.infrastructure.dictionary_loader import DictionaryLoader

Sym = namedtuple("Sym", ["text", "negated"])


def _real_red_flag_engine() -> RedFlagEngine:
    """محرّك مبنيّ من جدول الإنتاج نفسه — لا بيانات اختبار مصطنعة (C-3)."""
    return RedFlagEngine.from_dict(DictionaryLoader.load_red_flags())


def _targets(items):
    return [t for t, _ in items]


def test_core_oldcarts_for_primary_symptom():
    items = clinical.build_checklist([Sym("صداع", False)])
    targets = _targets(items)
    for slot in ("onset", "duration", "severity", "progression", "location",
                 "quality", "associated_symptoms", "aggravating", "relieving"):
        assert f"{slot}@صداع" in targets


def test_fever_adds_specific_slots():
    items = clinical.build_checklist([Sym("حرارة", False)])
    targets = _targets(items)
    assert "temperature@حرارة" in targets
    assert "chills@حرارة" in targets
    assert "cough@حرارة" in targets
    assert "sore_throat@حرارة" in targets


def test_abdominal_pain_adds_specific_slots():
    items = clinical.build_checklist([Sym("ألم في البطن", False)])
    targets = _targets(items)
    assert "migration@ألم في البطن" in targets
    assert "vomiting@ألم في البطن" in targets
    assert "diarrhea@ألم في البطن" in targets
    assert "constipation@ألم في البطن" in targets


def test_chest_pain_adds_specific_slots():
    items = clinical.build_checklist([Sym("ألم في الصدر", False)])
    targets = _targets(items)
    assert "radiation@ألم في الصدر" in targets
    assert "dyspnea@ألم في الصدر" in targets
    assert "sweating@ألم في الصدر" in targets
    assert "exertion@ألم في الصدر" in targets


# ----------------------------------------------------------------------
# FAST (facial_droop/unilateral_weakness/speech_difficulty) — أسئلة متابعة
# نوعية جديدة (لا تغيير على أي منطق تصنيف أو RedFlagEngine).
# ----------------------------------------------------------------------
def test_facial_droop_adds_fast_specific_slots():
    items = clinical.build_checklist([Sym("وجهي مايل", False)])
    targets = _targets(items)
    assert "onset_sudden@وجهي مايل" in targets
    assert "time_last_normal@وجهي مايل" in targets
    assert "side_affected@وجهي مايل" in targets


def test_time_last_normal_slot_present_for_unilateral_weakness():
    items = clinical.build_checklist([Sym("ضعف بنص جسمي", False)])
    targets = _targets(items)
    assert "time_last_normal@ضعف بنص جسمي" in targets


def test_speech_difficulty_adds_fast_specific_slots():
    items = clinical.build_checklist([Sym("كلامي متلعثم", False)])
    targets = _targets(items)
    assert "onset_sudden@كلامي متلعثم" in targets
    assert "time_last_normal@كلامي متلعثم" in targets
    assert "side_affected@كلامي متلعثم" in targets


def test_chest_pain_slots_unaffected_by_new_fast_rule():
    """انحدار: القاعدة الجديدة يجب ألّا تُغيّر سلوك ألم الصدر إطلاقاً."""
    items = clinical.build_checklist([Sym("ألم في الصدر", False)])
    targets = _targets(items)
    assert "radiation@ألم في الصدر" in targets
    assert "dyspnea@ألم في الصدر" in targets
    assert "sweating@ألم في الصدر" in targets
    assert "exertion@ألم في الصدر" in targets
    assert "onset_sudden@ألم في الصدر" not in targets


# ----------------------------------------------------------------------
# توسعة تغطية unilateral_weakness (تقرير الفحص: 3/11 → ~9/11) — عبارات
# مأخوذة حرفياً أو بأجزاء آمنة مُتحقَّق منها من red_flags.yaml، بلا تعميم
# "ما بقدر احرك" (قرار هندسي موثَّق بذاك الملف يرفضه صراحة).
# ----------------------------------------------------------------------
def test_move_hand_explicit_limb_gets_fast_slots():
    items = clinical.build_checklist([Sym("ما بقدر احرك ايدي", False)])
    targets = _targets(items)
    assert "onset_sudden@ما بقدر احرك ايدي" in targets
    assert "time_last_normal@ما بقدر احرك ايدي" in targets
    assert "side_affected@ما بقدر احرك ايدي" in targets


def test_move_leg_explicit_limb_gets_fast_slots():
    items = clinical.build_checklist([Sym("ما بقدر احرك رجلي", False)])
    targets = _targets(items)
    assert "onset_sudden@ما بقدر احرك رجلي" in targets
    assert "time_last_normal@ما بقدر احرك رجلي" in targets
    assert "side_affected@ما بقدر احرك رجلي" in targets


def test_attached_pronoun_form_gets_fast_slots():
    items = clinical.build_checklist([Sym("ما بقدر احركها", False)])
    targets = _targets(items)
    assert "onset_sudden@ما بقدر احركها" in targets
    assert "time_last_normal@ما بقدر احركها" in targets
    assert "side_affected@ما بقدر احركها" in targets


def test_sensory_loss_phrase_gets_fast_slots():
    items = clinical.build_checklist([Sym("ما بحس بايدي", False)])
    targets = _targets(items)
    assert "onset_sudden@ما بحس بايدي" in targets
    assert "time_last_normal@ما بحس بايدي" in targets
    assert "side_affected@ما بحس بايدي" in targets


def test_english_one_side_weakness_gets_fast_slots():
    items = clinical.build_checklist([Sym("one side weakness", False)])
    targets = _targets(items)
    assert "onset_sudden@one side weakness" in targets
    assert "time_last_normal@one side weakness" in targets
    assert "side_affected@one side weakness" in targets


def test_neck_movement_does_not_get_fast_slots():
    """الفحص الحرِج: 'ما بقدر احرك رقبتي' (عضلي-هيكلي، لا عصبي) يجب ألّا
    يُصنَّف FAST — القرار الهندسي بـred_flags.yaml (أسطر ١٢٦-١٢٨) يرفض
    تعميم 'ما بقدر احرك' وحدها لهذا السبب تحديداً، ولم يُستخدَم هنا."""
    items = clinical.build_checklist([Sym("ما بقدر احرك رقبتي", False)])
    targets = _targets(items)
    assert "onset_sudden@ما بقدر احرك رقبتي" not in targets
    assert "time_last_normal@ما بقدر احرك رقبتي" not in targets
    assert "side_affected@ما بقدر احرك رقبتي" not in targets


# ----------------------------------------------------------------------
# C-3: أولوية الأسئلة المرتبطة بعلم أحمر محتمل — آلية عامة، ليست خاصة بالصدر
# ----------------------------------------------------------------------
def test_chest_pain_specific_slots_precede_generic_oldcarts_with_red_flag_engine():
    """السيناريو المُثبَت: ألم الصدر شرط all_of لـHEALIX_REDFLAG_0001 —
    أسئلته النوعية (تنتشر/ضيق نفس/تعرّق/مجهود) يجب أن تتصدّر حتى قبل
    الأسئلة العامة (onset/duration/...) للشكوى نفسها."""
    items = clinical.build_checklist(
        [Sym("ألم صدر", False)], primary="ألم صدر",
        red_flag_engine=_real_red_flag_engine(),
    )
    targets = _targets(items)
    priority = {"radiation@ألم صدر", "dyspnea@ألم صدر", "sweating@ألم صدر", "exertion@ألم صدر"}
    core = {"onset@ألم صدر", "duration@ألم صدر", "severity@ألم صدر"}

    first_core_index = min(targets.index(t) for t in core)
    for t in priority:
        assert targets.index(t) < first_core_index, f"{t} لم يتصدّر OLDCARTS"


def test_thunderclap_headache_specific_slots_prioritized_proving_generic_mechanism():
    """إثبات العمومية (غير قلبي): نصّ يحمل شرطَي all_of معاً لـHEALIX_REDFLAG_0007
    (صداع + وصف رعدي "أسوأ صداع بحياتي") يجب أن تتصدّر أسئلته النوعية بنفس
    الآلية، بلا أي فحص خاص بالصدر بالكود (Phase 1.1 — بعد تصحيح C-3، النصّ
    يجب أن يحمل الشرطين معاً لا الصداع وحده)."""
    items = clinical.build_checklist(
        [Sym("صداع مفاجئ وشديد جدًا، أسوأ صداع بحياتي", False)],
        primary="صداع مفاجئ وشديد جدًا، أسوأ صداع بحياتي",
        red_flag_engine=_real_red_flag_engine(),
    )
    targets = _targets(items)
    symptom = "صداع مفاجئ وشديد جدًا، أسوأ صداع بحياتي"
    priority = {f"nausea@{symptom}", f"photophobia@{symptom}", f"neck_stiffness@{symptom}"}
    core = {f"onset@{symptom}", f"duration@{symptom}", f"severity@{symptom}"}

    first_core_index = min(targets.index(t) for t in core)
    for t in priority:
        assert targets.index(t) < first_core_index, f"{t} لم يتصدّر OLDCARTS"


def test_plain_headache_alone_is_not_prioritized_over_oldcarts():
    """Case 1 (Phase 1.1) — الصداع وحده (بلا وصف رعدي) لا يجوز أن يتصدّر:
    HEALIX_REDFLAG_0007 يتطلّب thunderclap_quality أيضاً، وهي غائبة هنا.
    هذا بالضبط السلوك الخاطئ الذي كانت الآلية القديمة تُنتجه."""
    items = clinical.build_checklist(
        [Sym("صداع", False)], primary="صداع",
        red_flag_engine=_real_red_flag_engine(),
    )
    targets = _targets(items)
    assert targets[0] == "onset@صداع", "لا يجوز ترقية أسئلة الصداع النوعية بمطابقة جزئية"


def test_without_red_flag_engine_order_is_unchanged():
    """التوافق الخلفي: بلا محرّك أعلام حمراء (المُعامِل الافتراضي None)،
    الترتيب القديم (OLDCARTS أولاً) يبقى كما كان تماماً."""
    items = clinical.build_checklist([Sym("ألم صدر", False)], primary="ألم صدر")
    targets = _targets(items)
    assert targets[0] == "onset@ألم صدر"


def test_negated_symptom_gets_no_specific_slots():
    items = clinical.build_checklist([Sym("حرارة", True)])
    targets = _targets(items)
    assert not any(t.startswith("temperature@") for t in targets)
    # لا عرَض مُثبَت → يبدأ بطلب وصف الشكوى.
    assert "chief_complaint" in targets


def test_context_slots_present():
    items = clinical.build_checklist([Sym("صداع", False)])
    targets = _targets(items)
    for slot in ("age", "gender", "chronic_diseases", "medications", "allergies", "smoking"):
        assert f"context:{slot}" in targets


def test_pregnancy_excluded_for_male():
    items = clinical.build_checklist([Sym("صداع", False)], context_text="ذكر")
    assert "context:pregnancy" not in _targets(items)


def test_pregnancy_included_when_gender_unknown():
    items = clinical.build_checklist([Sym("صداع", False)])
    assert "context:pregnancy" in _targets(items)


def test_next_missing_skips_covered():
    symptoms = [Sym("صداع", False)]
    asked = ["onset@صداع"]
    target, question = clinical.next_missing(symptoms, "", asked)
    assert target == "duration@صداع"
    assert isinstance(question, str) and question


def test_next_missing_returns_none_when_all_covered():
    symptoms = [Sym("صداع", False)]
    all_targets = [t for t, _ in clinical.build_checklist(symptoms)]
    assert clinical.next_missing(symptoms, "", all_targets) is None


def test_missing_targets_limit():
    symptoms = [Sym("صداع", False)]
    missing = clinical.missing_targets(symptoms, "", [], limit=3)
    assert len(missing) == 3


def test_male_is_never_asked_about_pregnancy():
    """انحدار: الذكر كان يُسأل عن الحمل بعد إزالة خريطة خانة→قيمة."""
    items = clinical.build_checklist([Sym("صداع", False)], context_text="مرحبا ٢٠ ذكر")
    assert "context:pregnancy" not in _targets(items)


def test_female_is_still_asked_about_pregnancy():
    items = clinical.build_checklist([Sym("صداع", False)], context_text="٢٥ أنثى")
    assert "context:pregnancy" in _targets(items)


def test_no_demographics_before_a_chief_complaint():
    """بلا أعراض: يجب طلب الشكوى الرئيسية فقط، لا العمر/الجنس/الحمل."""
    items = clinical.build_checklist([], context_text="مرحبا")
    assert _targets(items) == ["chief_complaint"]


def test_keeps_asking_chief_complaint_until_symptoms_arrive():
    """بلا أعراض: لا تنتهي المقابلة حتى لو سبق السؤال عن الشكوى."""
    item = clinical.next_missing([], "مرحبا", asked=["chief_complaint"])
    assert item is not None and item[0] == "chief_complaint"
    assert clinical.missing_targets([], "مرحبا", ["chief_complaint"]) == ["chief_complaint"]
