"""
اختبارات محرّك الأعلام الحمراء — **بصفر استدعاء LLM وصفر شبكة**.

هذا هو جوهر ضمان السلامة: المحرّك دوال خالصة، فيمكن اختباره استنفاداً.
لكل قاعدة: حالة موجبة، وحالة سالبة، وحالة قريبة-من-الحدّ تُثبت أنّ العتبة
ليست فضفاضة (مثلاً ألم الصدر وحده لا يكفي لإطلاق نمط ACS).
"""

import pytest

from app.domain.red_flag_engine import RedFlagEngine, normalize_arabic
from app.domain.red_flags import DetectionLayer, RiskLevel, escalate
from app.infrastructure.dictionary_loader import DictionaryLoader


@pytest.fixture(scope="module")
def engine() -> RedFlagEngine:
    """المحرّك مبنيّ من كتالوج الإنتاج نفسه — لا قواعد اختبار مصطنعة."""
    return RedFlagEngine.from_dict(DictionaryLoader.load_red_flags())


def fired(assessment) -> set:
    return {m.rule_id for m in assessment.matches}


# ----------------------------------------------------------------------
# التطبيع
# ----------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("ألم صَدر", "الم صدر"),
    ("أسوأ صداع", "اسوا صداع"),
    ("حرارة!!", "حراره"),          # التاء المربوطة تُوحَّد إلى هاء
    ("عمره ٤٠ يوم", "عمره 40 يوم"),
    ("ضيــــق تنفس", "ضيق تنفس"),
    ("Chest   Pain", "chest pain"),
])
def test_normalization(raw, expected):
    assert normalize_arabic(raw) == expected


def test_normalization_is_safe_on_empty():
    assert normalize_arabic("") == ""
    assert normalize_arabic(None or "") == ""


# ----------------------------------------------------------------------
# التصعيد أحادي الاتجاه (خاصية الأمان المركزية)
# ----------------------------------------------------------------------
def test_escalation_never_downgrades():
    assert escalate(RiskLevel.IMMEDIATE, RiskLevel.NONE) is RiskLevel.IMMEDIATE
    assert escalate(RiskLevel.NONE, RiskLevel.IMMEDIATE) is RiskLevel.IMMEDIATE
    assert escalate(RiskLevel.URGENT, RiskLevel.IMMEDIATE) is RiskLevel.IMMEDIATE
    assert escalate(RiskLevel.NONE, RiskLevel.NONE) is RiskLevel.NONE


# ----------------------------------------------------------------------
# ١) نمط ACS
# ----------------------------------------------------------------------
def test_acs_fires_on_chest_pain_with_dyspnea(engine):
    a = engine.evaluate_raw_text(["عندي ألم بصدري وضيق تنفس"])
    assert "HEALIX_REDFLAG_0001" in fired(a)
    assert a.risk_level is RiskLevel.IMMEDIATE
    assert a.is_emergency


def test_acs_fires_on_radiation_and_sweating(engine):
    a = engine.evaluate_raw_text(["وجع صدر شديد وعرق بارد والألم نازل بايدي"])
    assert "HEALIX_REDFLAG_0001" in fired(a)


def test_acs_does_not_fire_on_chest_pain_alone(engine):
    """العتبة ليست فضفاضة: ألم الصدر وحده لا يكفي."""
    a = engine.evaluate_raw_text(["عندي ألم بسيط بصدري"])
    assert "HEALIX_REDFLAG_0001" not in fired(a)


def test_acs_respects_explicit_negation(engine):
    a = engine.evaluate_raw_text(["ما عندي ألم صدر بس عندي رشح"])
    assert "HEALIX_REDFLAG_0001" not in fired(a)


# ----------------------------------------------------------------------
# ٢) السكتة / FAST — عتبة منخفضة عمداً
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "صار وجهي مايل فجأة",
    "عندي ضعف بنص جسمي",
    "صار كلامي متلعثم",
])
def test_stroke_fires_on_any_single_fast_sign(engine, text):
    a = engine.evaluate_raw_text([text])
    assert "HEALIX_REDFLAG_0002" in fired(a)
    assert a.risk_level is RiskLevel.IMMEDIATE


def test_stroke_does_not_fire_on_unrelated_text(engine):
    assert "HEALIX_REDFLAG_0002" not in fired(engine.evaluate_raw_text(["عندي زكام"]))


# ----------------------------------------------------------------------
# ٣) فقدان الوعي
# ----------------------------------------------------------------------
def test_loss_of_consciousness_fires_alone(engine):
    a = engine.evaluate_raw_text(["أغمي علي وأنا بالشغل"])
    assert "HEALIX_REDFLAG_0003" in fired(a)


# ----------------------------------------------------------------------
# ٤) التأق
# ----------------------------------------------------------------------
def test_anaphylaxis_fires_on_swelling_with_breathing(engine):
    a = engine.evaluate_raw_text(["لساني منتفخ وصار عندي ضيق تنفس"])
    assert "HEALIX_REDFLAG_0004" in fired(a)


def test_anaphylaxis_does_not_fire_on_rash_alone(engine):
    a = engine.evaluate_raw_text(["طلع عندي طفح جلدي بسيط"])
    assert "HEALIX_REDFLAG_0004" not in fired(a)


# ----------------------------------------------------------------------
# ٥) النزيف الشديد
# ----------------------------------------------------------------------
def test_severe_bleeding_requires_severity_qualifier(engine):
    assert "HEALIX_REDFLAG_0005" in fired(
        engine.evaluate_raw_text(["عندي نزيف شديد ما بوقف"]))
    assert "HEALIX_REDFLAG_0005" not in fired(
        engine.evaluate_raw_text(["نزيف بسيط من لثتي"]))


# ----------------------------------------------------------------------
# ٦) نزيف الحمل — بلا اشتراط ألم بطن (قرار سريري موثَّق)
# ----------------------------------------------------------------------
def test_pregnancy_bleeding_fires_without_abdominal_pain(engine):
    """النزيف غير المؤلم أثناء الحمل حالة طارئة — لا يُشترط الألم."""
    a = engine.evaluate_raw_text(["أنا حامل وصار عندي نزيف"])
    assert "HEALIX_REDFLAG_0006" in fired(a)
    assert a.risk_level is RiskLevel.IMMEDIATE


def test_pregnancy_bleeding_fires_across_turns(engine):
    """تركيب عبر الأدوار: الحمل ذُكر بدور والنزيف بدور لاحق."""
    a = engine.evaluate_raw_text(["أنا حامل بالشهر السابع", "عندي صداع", "صار في نزيف"])
    assert "HEALIX_REDFLAG_0006" in fired(a)


def test_bleeding_without_pregnancy_does_not_fire_rule_six(engine):
    assert "HEALIX_REDFLAG_0006" not in fired(engine.evaluate_raw_text(["عندي نزيف"]))


# ----------------------------------------------------------------------
# ٧) الصداع الرعدي
# ----------------------------------------------------------------------
def test_thunderclap_requires_quality_descriptor(engine):
    assert "HEALIX_REDFLAG_0007" in fired(
        engine.evaluate_raw_text(["صار عندي صداع وهو أسوأ صداع بحياتي"]))
    assert "HEALIX_REDFLAG_0007" not in fired(
        engine.evaluate_raw_text(["عندي صداع من امبارح"]))


# ----------------------------------------------------------------------
# ٨) حمى الرضيع — المستوى immediate (NICE NG143: fever >=38°C under 3
#    months is the highest/"red" urgency tier — re-graded from urgent,
#    كان تقييماً ناقصاً مقارنةً بالمصدر المُستشهَد به، انظر evidence_matrix.md RF-008)
# ----------------------------------------------------------------------
def test_infant_fever_fires_at_immediate(engine):
    a = engine.evaluate_raw_text(["ابني عمره شهرين وعنده حرارة"])
    assert "HEALIX_REDFLAG_0008" in fired(a)
    assert a.risk_level is RiskLevel.IMMEDIATE
    assert a.is_emergency


def test_adult_fever_does_not_fire(engine):
    assert "HEALIX_REDFLAG_0008" not in fired(engine.evaluate_raw_text(["عندي حرارة"]))


# ----------------------------------------------------------------------
# ٩) الأفكار الانتحارية
# ----------------------------------------------------------------------
def test_suicidal_ideation_fires(engine):
    a = engine.evaluate_raw_text(["تعبت وبفكر اقتل حالي"])
    assert "HEALIX_REDFLAG_0009" in fired(a)
    assert a.risk_level is RiskLevel.IMMEDIATE


def test_suicidal_action_message_is_supportive_and_directive(engine):
    a = engine.evaluate_raw_text(["بدي انهي حياتي"])
    action = a.primary_action_ar
    assert action and "لست وحدك" in action


# ----------------------------------------------------------------------
# ١٠) ضيق التنفّس الشديد
# ----------------------------------------------------------------------
def test_severe_dyspnea_requires_escalating_sign(engine):
    assert "HEALIX_REDFLAG_0010" in fired(
        engine.evaluate_raw_text(["عندي ضيق تنفس وما بقدر اكمل جملة"]))
    assert "HEALIX_REDFLAG_0010" not in fired(
        engine.evaluate_raw_text(["عندي ضيق تنفس خفيف مع المجهود"]))


# ----------------------------------------------------------------------
# الحالة السليمة والطبقة المنظَّمة
# ----------------------------------------------------------------------
def test_ordinary_complaint_raises_nothing(engine):
    a = engine.evaluate_raw_text(["عندي رشح وحرارة خفيفة من يومين"])
    assert a.matches == []
    assert a.risk_level is RiskLevel.NONE
    assert not a.is_emergency
    assert a.evaluated is True   # فُحص ولم يُوجد شيء ≠ لم يُفحص


def test_structured_layer_uses_positive_symptoms_only(engine):
    """الأعراض المنفية لا تُطلق قاعدة — النفي مقروء بنيوياً هنا."""
    a = engine.evaluate_record(
        symptom_texts_positive=["ألم صدر", "ضيق تنفس"], raw_texts=[])
    assert "HEALIX_REDFLAG_0001" in fired(a)
    assert a.matches[0].layer is DetectionLayer.STRUCTURED_RECORD

    b = engine.evaluate_record(symptom_texts_positive=["رشح"], raw_texts=[])
    assert b.matches == []


def test_structured_layer_still_sees_demographic_context_from_raw(engine):
    """قيد موثَّق: الحمل يُلتقط من الخام لأن السجل لا يحمل حقلاً له."""
    a = engine.evaluate_record(
        symptom_texts_positive=["نزيف"], raw_texts=["أنا حامل"])
    assert "HEALIX_REDFLAG_0006" in fired(a)


# ----------------------------------------------------------------------
# الدمج بين الطبقتين
# ----------------------------------------------------------------------
def test_merge_escalates_and_deduplicates(engine):
    l0 = engine.evaluate_raw_text(["عندي ألم صدر وضيق تنفس"])
    l2 = engine.evaluate_record(symptom_texts_positive=["ألم صدر", "ضيق تنفس"])
    merged = l0.merged_with(l2)

    assert merged.risk_level is RiskLevel.IMMEDIATE
    layers = {m.layer for m in merged.matches}
    assert layers == {DetectionLayer.RAW_TEXT, DetectionLayer.STRUCTURED_RECORD}


def test_merge_with_empty_preserves_emergency(engine):
    """الخاصية الحاسمة: نتيجة L0 تبقى حتى لو أعادت L2 لا شيء (فشل الـLLM)."""
    l0 = engine.evaluate_raw_text(["أغمي علي"])
    empty_l2 = engine.evaluate_record(symptom_texts_positive=[])
    merged = l0.merged_with(empty_l2)

    assert merged.risk_level is RiskLevel.IMMEDIATE
    assert "HEALIX_REDFLAG_0003" in fired(merged)


# ----------------------------------------------------------------------
# الشاهد وسلامة المخرَج
# ----------------------------------------------------------------------
def test_match_carries_patient_evidence_verbatim(engine):
    message = "عندي ألم بصدري وضيق تنفس من ساعة"
    a = engine.evaluate_raw_text([message])
    match = next(m for m in a.matches if m.rule_id == "HEALIX_REDFLAG_0001")
    assert match.evidence == message   # نصّ المريض كما ورد، بلا إعادة صياغة


def test_every_rule_has_actionable_arabic_guidance(engine):
    """تنبيه طوارئ بلا توجيه أسوأ من غياب القاعدة."""
    for rule in engine._rules:
        assert rule.recommended_action_ar.strip()
        assert rule.evidence_source.strip()


def test_no_rule_output_contains_a_diagnosis_claim(engine):
    """حدّ المجال: الإجراء يوجّه لرعاية، ولا يجزم بمرض."""
    for rule in engine._rules:
        assert "تشخيص" not in rule.recommended_action_ar
        assert "أنت مصاب" not in rule.recommended_action_ar


# ----------------------------------------------------------------------
# انحدارات: فجوات تغطية اكتُشفت بالاستقصاء وأُغلقت
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text, rule_id", [
    # ترتيب كلمات مختلف مع ضمير متّصل
    ("ايدي اليمين ما بقدر احركها", "HEALIX_REDFLAG_0002"),
    # كلمة متوسّطة تفصل المصطلح ("اجاني صداع فجأة وقوي")
    ("اجاني صداع فجأة وقوي", "HEALIX_REDFLAG_0007"),
    # صياغات حروف جرّ/ضمائر لألم الصدر
    ("حاسس بضغط بصدري ونفسي مقطوع", "HEALIX_REDFLAG_0001"),
    ("عندي وجع بالصدر ونازل على ايدي اليسار", "HEALIX_REDFLAG_0001"),
])
def test_known_phrasing_gaps_stay_closed(engine, text, rule_id):
    assert rule_id in fired(engine.evaluate_raw_text([text]))


def test_neck_immobility_does_not_fire_stroke(engine):
    """حماية من التوسّع المفرط: تيبّس الرقبة ليس علامة سكتة."""
    assert "HEALIX_REDFLAG_0002" not in fired(
        engine.evaluate_raw_text(["رقبتي متيبسة وما بقدر احرك رقبتي"]))


# ----------------------------------------------------------------------
# انحدار حرج: L0 وحده يجب أن يكشف نمط ACS بصياغاته الشائعة
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    # «ضاغط» تفصل بين "ألم" و"بصدري" — كانت تُفشل L0 كاملاً
    "عندي ألم ضاغط بصدري وضيق نفس وعرق بارد والألم نازل بإيدي اليسار",
    "حاسس بثقل على صدري ومعي ضيق تنفس",
    "في ضغط بصدري وعرق بارد",
])
def test_l0_alone_detects_acs_phrasings(engine, text):
    """حاسم: لو انهار الـLLM فهذه هي طبقة الأمان الوحيدة المتبقّية."""
    assessment = engine.evaluate_raw_text([text])
    assert "HEALIX_REDFLAG_0001" in fired(assessment)
    assert assessment.risk_level is RiskLevel.IMMEDIATE


# ----------------------------------------------------------------------
# C-3 (Phase 1.1 — تصحيح دلالي): استعلام أولوية الأسئلة — قراءة فقط، لا
# يُطلق أي قاعدة. الاسم القديم ``is_potential_trigger`` حُذف: كان يكتفي
# بمطابقة مجموعة واحدة فقط من ``all_of`` قاعدة قد تتطلّب أكثر من مجموعة معاً
# (خطأ مُثبَت بتدقيق Phase 0) — ``is_red_flag_pathway_evidence`` يتطلّب
# تحقّق **كل** مجموعات ``all_of`` معاً، أو عضواً من ``any_of`` حين لا توجد
# ``all_of`` إطلاقاً (مثال FAST).
# ----------------------------------------------------------------------
def test_chest_pain_is_pathway_evidence_single_all_of_group(engine):
    """ألم الصدر: شرط all_of وحيد لـHEALIX_REDFLAG_0001 — يتحقّق بهذا النصّ
    وحده تلقائياً (Case 6)."""
    assert engine.is_red_flag_pathway_evidence("ألم صدر") is True


def test_headache_alone_is_not_pathway_evidence_missing_thunderclap(engine):
    """Case 1 — الصداع وحده لا يكفي: HEALIX_REDFLAG_0007 يتطلّب headache
    و thunderclap_quality معاً؛ مطابقة headache فقط لا تعني أنّ القاعدة على
    مسار الإطلاق. هذا هو الخطأ الدلالي الذي أصلحه Phase 1.1 تحديداً — كان
    الاستعلام القديم يعيد True هنا خطأً."""
    assert engine.is_red_flag_pathway_evidence("عندي صداع") is False


def test_thunderclap_headache_is_pathway_evidence_both_groups_present(engine):
    """Case 2 — نصّ واحد يحمل كلا مجموعتَي all_of معاً (headache +
    thunderclap_quality) يجب أن يُعَدّ دليل مسار فعلياً."""
    assert engine.is_red_flag_pathway_evidence(
        "صداع مفاجئ وشديد جدًا، أسوأ صداع بحياتي"
    ) is True


def test_fever_alone_is_not_pathway_evidence_missing_infant_context(engine):
    """Case 3 — الحرارة وحدها لا تكفي: HEALIX_REDFLAG_0008 يتطلّب
    infant_context و fever معاً؛ لا سياق رضيع هنا."""
    assert engine.is_red_flag_pathway_evidence("عندي حرارة") is False


def test_bleeding_alone_is_pathway_evidence_via_standalone_rule_not_pregnancy(engine):
    """Case 4 — 'نزيف' وحدها **تبقى** True، لكن بسبب قاعدة مستقلّة حقيقية
    (HEALIX_REDFLAG_0005: all_of=['bleeding'] وحدها) لا بسبب قاعدة الحمل
    (HEALIX_REDFLAG_0006 تتطلّب bleeding+pregnancy_context معاً). الاختبار
    التالي يعزل قاعدة الحمل تحديداً لإثبات أنها لا تُساهم بالخطأ القديم."""
    assert engine.is_red_flag_pathway_evidence("عندي نزيف") is True


def test_pregnancy_context_alone_is_not_pathway_evidence_missing_bleeding(engine):
    """Case 4 (عزل قاعدة الحمل تحديداً) — 'أنا حامل' وحدها بلا نزيف لا يجوز
    أن تُعَدّ دليل مسار لـHEALIX_REDFLAG_0006 (يتطلّب bleeding أيضاً)، ولا
    تلمس أي قاعدة أخرى."""
    assert engine.is_red_flag_pathway_evidence("انا حامل") is False


def test_facial_droop_is_pathway_evidence_any_of_only_rule(engine):
    """Case 5 — HEALIX_REDFLAG_0002 (FAST) بلا all_of إطلاقاً؛ عضو واحد من
    any_of (تدلّي الوجه) كافٍ — كانت الآلية القديمة تُغفل هذه الحالة كلياً
    لأنّها فحصت all_of فقط."""
    assert engine.is_red_flag_pathway_evidence("وجهي مايل") is True


def test_unrelated_text_is_not_pathway_evidence(engine):
    assert engine.is_red_flag_pathway_evidence("عندي رشح خفيف") is False


def test_pathway_evidence_query_does_not_fire_any_rule():
    """قراءة فقط: لا يُنتج RedFlagAssessment ولا يُغيّر حالة المحرّك."""
    engine = RedFlagEngine.from_dict(DictionaryLoader.load_red_flags())
    engine.is_red_flag_pathway_evidence("ألم صدر")
    # نفس المحرّك يُقيَّم بعدها بشكل طبيعي تماماً — لا أثر جانبي.
    assessment = engine.evaluate_raw_text(["عندي رشح خفيف"])
    assert assessment.risk_level is RiskLevel.NONE
