"""
Healix - Dictionary Loader
تحميل قواميس البيانات الخارجية (قواعد التحقّق، مخطّطات الترميز، قاموس
التخصّصات) مرّة واحدة عند بدء التشغيل، مع التحقّق من بنيتها فوراً — فشل
الإقلاع بوضوح لو كانت مشوّهة، بدل اكتشاف الخلل لاحقاً وسط معالجة طلب مريض
حقيقي.

طبقة بنية تحتية خالصة (I/O + تحقّق بنيوي) — لا منطق تحقّق طبي أو ترميز أو
توصية هنا؛ تلك مسؤولية ``domain.feature_validator.FeatureValidator``،
``domain.feature_encoder.FeatureEncoder``، و
``domain.rule_based_specialty_recommender.RuleBasedSpecialtyRecommender``
اللذين يستقبلون البيانات المُحمَّلة جاهزة عبر الحقن (Dependency Injection)،
بلا أي قراءة ملفات لحالهم.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.exceptions import FeatureValidationError

_NUMERIC_TYPES = ("int", "float")
_VALID_OUT_OF_RANGE_STRATEGIES = ("reject", "clip")

_DICTIONARIES_DIR = Path(__file__).resolve().parent.parent / "dictionaries"
_DEFAULT_RULES_PATH = _DICTIONARIES_DIR / "feature_validation_rules.json"
_DEFAULT_SCHEMA_PATH = _DICTIONARIES_DIR / "feature_schemas" / "v1.json"
_DEFAULT_SPECIALTY_LOOKUP_PATH = _DICTIONARIES_DIR / "specialty_lookup.yaml"
_DEFAULT_RED_FLAGS_PATH = _DICTIONARIES_DIR / "red_flags.yaml"
_DEFAULT_CLINICAL_PRIORITY_PATH = _DICTIONARIES_DIR / "clinical_priority.yaml"
_DEFAULT_SYMPTOM_EVIDENCE_MAP_PATH = _DICTIONARIES_DIR / "symptom_evidence_map.yaml"


class DictionaryLoader:
    """يحمّل قواميس JSON مرّة واحدة، ويتحقّق من بنيتها فوراً."""

    @staticmethod
    def load_feature_validation_rules(path: Optional[Path] = None) -> Dict[str, Any]:
        """يحمّل قاموس قواعد التحقّق من الميزات ويتحقّق من بنيته.

        يرفع ``FeatureValidationError`` عند تعذّر القراءة، أو JSON غير صالح،
        أو بنية لا تطابق العقد المتوقَّع — بوضوح، عند الإقلاع، لا أثناء طلب.
        """
        data = DictionaryLoader._load_json(path or _DEFAULT_RULES_PATH)
        _validate_validation_rules_structure(data, path or _DEFAULT_RULES_PATH)
        return data

    @staticmethod
    def load_feature_schema(path: Optional[Path] = None) -> Dict[str, Any]:
        """يحمّل مخطّط ترميز الميزات (Phase 3.3) ويتحقّق من بنيته وتماسكه
        الداخلي (تطابق feature_order مع الحقول المُعرَّفة فعلياً).

        يرفع ``FeatureValidationError`` عند تعذّر القراءة، أو JSON غير صالح،
        أو بنية/تماسك لا يطابق العقد المتوقَّع — بوضوح، عند الإقلاع.
        """
        resolved_path = path or _DEFAULT_SCHEMA_PATH
        data = DictionaryLoader._load_json(resolved_path)
        _validate_feature_schema_structure(data, resolved_path)
        return data

    @staticmethod
    def load_specialty_lookup(path: Optional[Path] = None) -> Dict[str, Any]:
        """يحمّل قاموس (مرض → تخصّص) من YAML (Phase 3.6) ويتحقّق من بنيته.

        يرفع ``FeatureValidationError`` عند تعذّر القراءة، أو YAML غير صالح،
        أو بنية لا تطابق العقد المتوقَّع — بوضوح، عند الإقلاع، لا أثناء طلب.
        """
        resolved_path = path or _DEFAULT_SPECIALTY_LOOKUP_PATH
        data = DictionaryLoader._load_yaml(resolved_path)
        _validate_specialty_lookup_structure(data, resolved_path)
        return data

    @staticmethod
    def load_red_flags(path: Optional[Path] = None) -> Dict[str, Any]:
        """يحمّل كتالوج الأعلام الحمراء من YAML ويتحقّق من بنيته وتماسكه.

        التحقّق هنا أشدّ من بقيّة القواميس عن قصد: قاعدة مشوّهة أو تشير إلى
        مجموعة مصطلحات غير معرّفة تعني **فشل كشف حالة طارئة صامتاً**. لذلك
        يفشل الإقلاع بوضوح بدل تمرير كتالوج ناقص إلى الإنتاج.
        """
        resolved_path = path or _DEFAULT_RED_FLAGS_PATH
        data = DictionaryLoader._load_yaml(resolved_path)
        _validate_red_flags_structure(data, resolved_path)
        return data

    @staticmethod
    def load_clinical_priority(path: Optional[Path] = None) -> Dict[str, Any]:
        """يحمّل جدول الأولوية السريرية من YAML ويتحقّق من بنيته.

        جدول مشوّه يعني ترتيباً سريرياً خاطئاً للأسئلة — لذلك يفشل الإقلاع
        بوضوح بدل تمرير جدول ناقص إلى الإنتاج.
        """
        resolved_path = path or _DEFAULT_CLINICAL_PRIORITY_PATH
        data = DictionaryLoader._load_yaml(resolved_path)
        _validate_clinical_priority_structure(data, resolved_path)
        return data

    @staticmethod
    def load_symptom_evidence_map(path: Optional[Path] = None) -> Dict[str, Any]:
        """يحمّل قاموس ربط الأعراض العربية برموز أدلة DDXPlus ويتحقّق من بنيته.

        جدول مشوّه يعني متجه أدلة خاطئاً يُغذّي مُتنبِّئ ML — لذلك يفشل
        الإقلاع بوضوح بدل تمرير قاموس ربط ناقص إلى الإنتاج.
        """
        resolved_path = path or _DEFAULT_SYMPTOM_EVIDENCE_MAP_PATH
        data = DictionaryLoader._load_yaml(resolved_path)
        _validate_symptom_evidence_map_structure(data, resolved_path)
        return data

    @staticmethod
    def _read_text(path: Path) -> str:
        """قراءة نصّ خام — مشتركة بين كل محمِّلات القواميس (JSON وYAML)."""
        if not path.exists():
            raise FeatureValidationError(f"الملف غير موجود: {path}")

        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FeatureValidationError(f"تعذّرت قراءة الملف ({path}): {exc}") from exc

    @staticmethod
    def _load_json(path: Path) -> Any:
        """قراءة + تحليل JSON."""
        raw_text = DictionaryLoader._read_text(path)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise FeatureValidationError(f"الملف ليس JSON صالحاً ({path}): {exc}") from exc

    @staticmethod
    def _load_yaml(path: Path) -> Any:
        """قراءة + تحليل YAML."""
        raw_text = DictionaryLoader._read_text(path)
        try:
            return yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise FeatureValidationError(f"الملف ليس YAML صالحاً ({path}): {exc}") from exc


def _validate_validation_rules_structure(data: Any, source: Path) -> None:
    """تحقّق بنيوي صارم — يفشل الإقلاع بوضوح عند أي انحراف عن العقد المتوقَّع."""
    if not isinstance(data, dict):
        raise FeatureValidationError(f"جذر قاموس قواعد التحقّق يجب أن يكون كائناً ({source}).")

    fields = data.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise FeatureValidationError(
            f"قاموس قواعد التحقّق يجب أن يحتوي 'fields' غير فارغ ({source})."
        )

    for field_name, spec in fields.items():
        if not isinstance(spec, dict):
            raise FeatureValidationError(
                f"قاعدة الحقل '{field_name}' يجب أن تكون كائناً ({source})."
            )

        field_type = spec.get("type")

        if field_type in _NUMERIC_TYPES:
            _validate_numeric_spec(field_name, spec, source)
        elif field_type == "enum":
            _validate_enum_spec(field_name, spec, source)
        else:
            raise FeatureValidationError(
                f"الحقل '{field_name}': نوع غير مدعوم '{field_type}' ({source})."
            )


def _validate_numeric_spec(field_name: str, spec: Dict[str, Any], source: Path) -> None:
    minimum = spec.get("min")
    maximum = spec.get("max")
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        raise FeatureValidationError(
            f"الحقل '{field_name}' من نوع '{spec.get('type')}' يتطلّب min/max رقميين ({source})."
        )
    if minimum > maximum:
        raise FeatureValidationError(f"الحقل '{field_name}': min أكبر من max ({source}).")

    strategy = spec.get("on_out_of_range", "reject")
    if strategy not in _VALID_OUT_OF_RANGE_STRATEGIES:
        raise FeatureValidationError(
            f"الحقل '{field_name}': on_out_of_range غير مدعومة '{strategy}' ({source})."
        )


def _validate_enum_spec(field_name: str, spec: Dict[str, Any], source: Path) -> None:
    allowed = spec.get("allowed")
    if not isinstance(allowed, list) or not allowed:
        raise FeatureValidationError(
            f"الحقل '{field_name}' من نوع 'enum' يتطلّب 'allowed' غير فارغة ({source})."
        )


# ----------------------------------------------------------------------
# تحقّق بنيوي لمخطّط ترميز الميزات (Phase 3.3)
# ----------------------------------------------------------------------
def _validate_feature_schema_structure(data: Any, source: Path) -> None:
    """تحقّق بنيوي + تماسك داخلي صارم لمخطّط الترميز — يفشل الإقلاع بوضوح.

    التماسك الحاسم: ``feature_order`` يجب أن يطابق تماماً (بلا نقص ولا زيادة
    ولا تكرار) كل الأعمدة التي سينتجها الترميز فعلياً — أعمدة numeric_fields
    وboolean_fields كما هي، وأعمدة categorical_fields بعد توسيع one-hot.
    """
    if not isinstance(data, dict):
        raise FeatureValidationError(f"جذر مخطّط الترميز يجب أن يكون كائناً ({source}).")

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise FeatureValidationError(f"مخطّط الترميز يتطلّب 'schema_version' نصّياً غير فارغ ({source}).")

    feature_order = data.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order or not all(
        isinstance(name, str) for name in feature_order
    ):
        raise FeatureValidationError(
            f"مخطّط الترميز يتطلّب 'feature_order' قائمة نصوص غير فارغة ({source})."
        )

    if len(feature_order) != len(set(feature_order)):
        raise FeatureValidationError(f"مخطّط الترميز: 'feature_order' يحوي أسماء مكرّرة ({source}).")

    numeric_fields = data.get("numeric_fields", {})
    boolean_fields = data.get("boolean_fields", {})
    categorical_fields = data.get("categorical_fields", {})

    if not numeric_fields and not boolean_fields and not categorical_fields:
        raise FeatureValidationError(
            f"مخطّط الترميز يتطلّب حقلاً واحداً على الأقل (numeric/boolean/categorical) ({source})."
        )

    expected_columns: List[str] = []

    for name, spec in _require_field_dict(numeric_fields, "numeric_fields", source).items():
        _validate_scalar_field_spec(name, spec, "numeric_fields", source)
        expected_columns.append(name)

    for name, spec in _require_field_dict(boolean_fields, "boolean_fields", source).items():
        _validate_scalar_field_spec(name, spec, "boolean_fields", source)
        expected_columns.append(name)

    for name, spec in _require_field_dict(categorical_fields, "categorical_fields", source).items():
        expected_columns.extend(_validate_categorical_field_spec(name, spec, source))

    if len(expected_columns) != len(set(expected_columns)):
        raise FeatureValidationError(
            f"مخطّط الترميز: تعارض أسماء أعمدة بين الحقول المُعرَّفة ({source})."
        )

    if set(expected_columns) != set(feature_order):
        missing = sorted(set(expected_columns) - set(feature_order))
        extra = sorted(set(feature_order) - set(expected_columns))
        raise FeatureValidationError(
            f"مخطّط الترميز: 'feature_order' لا يطابق الأعمدة المُعرَّفة فعلياً "
            f"({source}) — ناقصة: {missing} | زائدة: {extra}."
        )


def _require_field_dict(value: Any, key: str, source: Path) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise FeatureValidationError(f"مخطّط الترميز: '{key}' يجب أن يكون كائناً ({source}).")
    return value


def _validate_scalar_field_spec(field_name: str, spec: Any, section: str, source: Path) -> None:
    if not isinstance(spec, dict):
        raise FeatureValidationError(
            f"مخطّط الترميز: حقل '{field_name}' بقسم '{section}' يجب أن يكون كائناً ({source})."
        )
    if not isinstance(spec.get("nullable"), bool):
        raise FeatureValidationError(
            f"مخطّط الترميز: حقل '{field_name}' بقسم '{section}' يتطلّب 'nullable' منطقياً ({source})."
        )


def _validate_categorical_field_spec(field_name: str, spec: Any, source: Path) -> List[str]:
    if not isinstance(spec, dict):
        raise FeatureValidationError(
            f"مخطّط الترميز: حقل فئوي '{field_name}' يجب أن يكون كائناً ({source})."
        )
    if not isinstance(spec.get("nullable"), bool):
        raise FeatureValidationError(
            f"مخطّط الترميز: حقل فئوي '{field_name}' يتطلّب 'nullable' منطقياً ({source})."
        )

    values = spec.get("values")
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        raise FeatureValidationError(
            f"مخطّط الترميز: حقل فئوي '{field_name}' يتطلّب 'values' قائمة نصوص غير فارغة ({source})."
        )
    if len(values) != len(set(values)):
        raise FeatureValidationError(
            f"مخطّط الترميز: حقل فئوي '{field_name}' يحوي 'values' مكرّرة ({source})."
        )

    prefix = spec.get("one_hot_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise FeatureValidationError(
            f"مخطّط الترميز: حقل فئوي '{field_name}' يتطلّب 'one_hot_prefix' نصّياً غير فارغ ({source})."
        )

    return [f"{prefix}_{value}" for value in values]


# ----------------------------------------------------------------------
# تحقّق بنيوي لقاموس (مرض → تخصّص) (Phase 3.6)
# ----------------------------------------------------------------------
def _validate_specialty_lookup_structure(data: Any, source: Path) -> None:
    """تحقّق بنيوي صارم — يفشل الإقلاع بوضوح عند أي انحراف عن العقد المتوقَّع.

    العقد: كائن جذر غير فارغ، كل مفتاح اسم مرض (نصّ)، وكل قيمة كائن يحوي
    'specialty' نصّياً غير فارغ.
    """
    if not isinstance(data, dict) or not data:
        raise FeatureValidationError(
            f"قاموس التخصّصات يجب أن يكون كائناً غير فارغ ({source})."
        )

    for disease_name, spec in data.items():
        if not isinstance(disease_name, str) or not disease_name.strip():
            raise FeatureValidationError(
                f"قاموس التخصّصات: اسم مرض غير صالح ({source})."
            )
        if not isinstance(spec, dict):
            raise FeatureValidationError(
                f"قاموس التخصّصات: قيمة '{disease_name}' يجب أن تكون كائناً ({source})."
            )
        specialty = spec.get("specialty")
        if not isinstance(specialty, str) or not specialty.strip():
            raise FeatureValidationError(
                f"قاموس التخصّصات: '{disease_name}' يتطلّب 'specialty' نصّياً غير فارغ ({source})."
            )


def _validate_red_flags_structure(data: Any, source: Path) -> None:
    """تحقّق بنيوي صارم لكتالوج الأعلام الحمراء.

    العقد: كائن جذر يحوي ``term_groups`` (اسم → قائمة مصطلحات نصّية غير
    فارغة) و``rules`` (قائمة غير فارغة). كل قاعدة تتطلّب معرّفاً فريداً،
    اسمين، مستوى خطورة ونوع مُطلِق ضمن التعدادات المعروفة، شرطاً واحداً على
    الأقل، وإجراءً موصى به بالعربية — فإجراء فارغ يعني تنبيه طوارئ بلا
    توجيه، وهو أسوأ من غياب القاعدة.

    كل مجموعة مذكورة في قاعدة يجب أن تكون معرّفة فعلاً: مرجع مكسور يعني
    قاعدة لا تُطلق أبداً — فشل صامت بالضبط ما نمنعه هنا.
    """
    from app.domain.red_flags import RiskLevel, TriggerType

    if not isinstance(data, dict):
        raise FeatureValidationError(f"كتالوج الأعلام الحمراء: الجذر يجب أن يكون كائناً ({source}).")

    groups = data.get("term_groups")
    if not isinstance(groups, dict) or not groups:
        raise FeatureValidationError(
            f"كتالوج الأعلام الحمراء: 'term_groups' مطلوب وغير فارغ ({source})."
        )
    for name, terms in groups.items():
        if not isinstance(terms, list) or not terms:
            raise FeatureValidationError(
                f"كتالوج الأعلام الحمراء: المجموعة '{name}' يجب أن تكون قائمة غير فارغة ({source})."
            )
        for term in terms:
            if not isinstance(term, str) or not term.strip():
                raise FeatureValidationError(
                    f"كتالوج الأعلام الحمراء: المجموعة '{name}' تحوي مصطلحاً فارغاً ({source})."
                )

    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise FeatureValidationError(
            f"كتالوج الأعلام الحمراء: 'rules' مطلوب وغير فارغ ({source})."
        )

    valid_levels = {level.value for level in RiskLevel}
    valid_triggers = {trigger.value for trigger in TriggerType}
    seen_ids: set = set()

    for rule in rules:
        if not isinstance(rule, dict):
            raise FeatureValidationError(
                f"كتالوج الأعلام الحمراء: كل قاعدة يجب أن تكون كائناً ({source})."
            )
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise FeatureValidationError(
                f"كتالوج الأعلام الحمراء: 'rule_id' نصّي غير فارغ مطلوب لكل قاعدة ({source})."
            )
        if rule_id in seen_ids:
            raise FeatureValidationError(
                f"كتالوج الأعلام الحمراء: معرّف مكرّر '{rule_id}' ({source})."
            )
        seen_ids.add(rule_id)

        for key in ("name_ar", "name_en", "recommended_action_ar", "evidence_source"):
            value = rule.get(key)
            if not isinstance(value, str) or not value.strip():
                raise FeatureValidationError(
                    f"كتالوج الأعلام الحمراء: '{rule_id}' يتطلّب '{key}' نصّياً غير فارغ ({source})."
                )

        if rule.get("risk_level") not in valid_levels:
            raise FeatureValidationError(
                f"كتالوج الأعلام الحمراء: '{rule_id}' مستوى خطورة غير معروف "
                f"'{rule.get('risk_level')}' (المسموح: {sorted(valid_levels)}) ({source})."
            )
        if rule.get("trigger_type") not in valid_triggers:
            raise FeatureValidationError(
                f"كتالوج الأعلام الحمراء: '{rule_id}' نوع مُطلِق غير معروف "
                f"'{rule.get('trigger_type')}' (المسموح: {sorted(valid_triggers)}) ({source})."
            )

        all_of = rule.get("all_of") or []
        any_of = rule.get("any_of") or []
        if not isinstance(all_of, list) or not isinstance(any_of, list):
            raise FeatureValidationError(
                f"كتالوج الأعلام الحمراء: '{rule_id}' — 'all_of' و'any_of' يجب أن تكونا قائمتين ({source})."
            )
        if not all_of and not any_of:
            raise FeatureValidationError(
                f"كتالوج الأعلام الحمراء: '{rule_id}' بلا أي شرط — لن تُطلق أبداً ({source})."
            )
        for group in list(all_of) + list(any_of):
            if group not in groups:
                raise FeatureValidationError(
                    f"كتالوج الأعلام الحمراء: '{rule_id}' يشير إلى مجموعة غير معرّفة "
                    f"'{group}' — القاعدة لن تُطلق أبداً ({source})."
                )


def _validate_clinical_priority_structure(data: Any, source: Path) -> None:
    """تحقّق بنيوي لجدول الأولوية السريرية.

    العقد: ``concepts`` غير فارغ، كل مفهوم يحمل ``acuity`` رقمياً ضمن
    0..100 و``terms`` قائمة نصوص غير فارغة. ``default_acuity`` رقم.
    قيمة acuity خارج المدى تعني ترتيباً منحرفاً بصمت، فتُرفض عند الإقلاع.
    """
    if not isinstance(data, dict):
        raise FeatureValidationError(
            f"جدول الأولوية السريرية: الجذر يجب أن يكون كائناً ({source})."
        )

    default_acuity = data.get("default_acuity", 30)
    if not isinstance(default_acuity, (int, float)) or not 0 <= default_acuity <= 100:
        raise FeatureValidationError(
            f"جدول الأولوية السريرية: 'default_acuity' رقم بين 0 و100 ({source})."
        )

    concepts = data.get("concepts")
    if not isinstance(concepts, dict) or not concepts:
        raise FeatureValidationError(
            f"جدول الأولوية السريرية: 'concepts' مطلوب وغير فارغ ({source})."
        )

    for name, spec in concepts.items():
        if not isinstance(spec, dict):
            raise FeatureValidationError(
                f"جدول الأولوية السريرية: المفهوم '{name}' يجب أن يكون كائناً ({source})."
            )
        acuity = spec.get("acuity")
        if not isinstance(acuity, (int, float)) or not 0 <= acuity <= 100:
            raise FeatureValidationError(
                f"جدول الأولوية السريرية: '{name}' يتطلّب 'acuity' رقماً بين 0 و100 "
                f"(الموجود: {acuity!r}) ({source})."
            )
        terms = spec.get("terms")
        if not isinstance(terms, list) or not terms:
            raise FeatureValidationError(
                f"جدول الأولوية السريرية: '{name}' يتطلّب 'terms' قائمة غير فارغة ({source})."
            )
        for term in terms:
            if not isinstance(term, str) or not term.strip():
                raise FeatureValidationError(
                    f"جدول الأولوية السريرية: '{name}' يحوي مصطلحاً فارغاً ({source})."
                )

    for key in ("severity_terms", "emphasis_terms"):
        value = data.get(key)
        if value is None:
            continue
        if key == "severity_terms" and not isinstance(value, dict):
            raise FeatureValidationError(
                f"جدول الأولوية السريرية: '{key}' يجب أن يكون كائناً ({source})."
            )
        if key == "emphasis_terms" and not isinstance(value, list):
            raise FeatureValidationError(
                f"جدول الأولوية السريرية: '{key}' يجب أن يكون قائمة ({source})."
            )


def _validate_symptom_evidence_map_structure(data: Any, source: Path) -> None:
    """تحقّق بنيوي صارم — يفشل الإقلاع بوضوح عند أي انحراف عن العقد المتوقَّع."""
    if not isinstance(data, dict):
        raise FeatureValidationError(f"جذر قاموس ربط الأعراض بالأدلة يجب أن يكون كائناً ({source}).")

    mappings = data.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise FeatureValidationError(f"قاموس ربط الأعراض بالأدلة يتطلّب 'mappings' قائمة غير فارغة ({source}).")

    for entry in mappings:
        if not isinstance(entry, dict):
            raise FeatureValidationError(f"عنصر ربط أعراض غير صالح (ليس كائناً) ({source}).")

        concept = entry.get("concept")
        if not isinstance(concept, str) or not concept.strip():
            raise FeatureValidationError(f"عنصر ربط أعراض يتطلّب 'concept' نصّياً غير فارغ ({source}).")

        names = entry.get("names")
        if not isinstance(names, list) or not names or not all(
            isinstance(n, str) and n.strip() for n in names
        ):
            raise FeatureValidationError(
                f"'{concept}': يتطلّب 'names' قائمة نصوص غير فارغة ({source})."
            )

        codes = entry.get("evidence_codes")
        if not isinstance(codes, list) or not codes or not all(
            isinstance(c, str) and c.strip() for c in codes
        ):
            raise FeatureValidationError(
                f"'{concept}': يتطلّب 'evidence_codes' قائمة نصوص غير فارغة ({source})."
            )
