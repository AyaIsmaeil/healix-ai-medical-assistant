"""اختبارات وحدة للمعرفة السريرية والمُخطِّط (domain.clinical)."""

from collections import namedtuple

from app.domain import clinical

Sym = namedtuple("Sym", ["text", "negated"])


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
