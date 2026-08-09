"""
Healix - Model Loader
تحميل آثار نموذج ML المدرَّب (models/) مرّة واحدة عند بدء التشغيل، مع التحقّق
من بنيتها وتماسكها فوراً — فشل الإقلاع بوضوح لو كانت مفقودة أو غير متطابقة،
بدل اكتشاف الخلل لاحقاً وسط معالجة طلب مريض حقيقي. نفس فلسفة
``infrastructure.dictionary_loader.DictionaryLoader`` بالضبط.

طبقة بنية تحتية خالصة (I/O + تحقّق بنيوي) — لا منطق استنتاج هنا؛ تلك مسؤولية
``domain.ml_disease_predictor.MLDiseasePredictor`` الذي يستقبل النتائج جاهزة
عبر الحقن (Dependency Injection)، بلا أي قراءة ملفات لحاله.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib

from app.exceptions import ModelLoadError

EXPECTED_N_FEATURES = 225
EXPECTED_N_CLASSES = 49

# models/ شقيق app/ (لا داخله) — ثلاث مستويات .parent من هذا الملف، بخلاف
# DictionaryLoader (مستويين، حيث dictionaries/ داخل app/ نفسها).
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_DEFAULT_MODEL_PATH = _MODELS_DIR / "xgboost_model.pkl"
_DEFAULT_FEATURE_NAMES_PATH = _MODELS_DIR / "feature_names.json"
_DEFAULT_LABEL_ENCODER_PATH = _MODELS_DIR / "label_encoder.joblib"


class ModelLoader:
    """يحمّل آثار النموذج المحفوظة، ويتحقّق من بنيتها وتطابقها فوراً."""

    @staticmethod
    def load_model(path: Optional[Path] = None) -> Any:
        """يحمّل النموذج المدرَّب (joblib/pickle). يرفع ``ModelLoadError`` عند
        غياب الملف أو تعذّر التحميل أو غياب ``predict_proba``."""
        resolved_path = path or _DEFAULT_MODEL_PATH
        if not resolved_path.exists():
            raise ModelLoadError(f"ملف النموذج غير موجود: {resolved_path}")

        try:
            model = joblib.load(resolved_path)
        except Exception as exc:  # noqa: BLE001 - أي فشل تحميل يُغلَّف بنوع خطأ الدومين
            raise ModelLoadError(f"تعذّر تحميل النموذج ({resolved_path}): {exc}") from exc

        if not hasattr(model, "predict_proba"):
            raise ModelLoadError(
                f"الكائن المحمَّل من {resolved_path} لا يوفّر predict_proba — ليس مصنِّفاً صالحاً."
            )
        return model

    @staticmethod
    def load_feature_names(path: Optional[Path] = None) -> List[str]:
        """يحمّل قائمة أسماء الأعمدة المُرمَّزة (ترتيبها هو ترتيب مدخل النموذج).
        يرفع ``ModelLoadError`` عند غياب الملف أو JSON غير صالح أو بنية خاطئة."""
        resolved_path = path or _DEFAULT_FEATURE_NAMES_PATH
        if not resolved_path.exists():
            raise ModelLoadError(f"ملف feature_names غير موجود: {resolved_path}")

        try:
            data = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"تعذّرت قراءة feature_names ({resolved_path}): {exc}") from exc

        if not isinstance(data, list) or not data or not all(isinstance(name, str) for name in data):
            raise ModelLoadError(f"feature_names يجب أن يكون قائمة نصوص غير فارغة ({resolved_path}).")
        return data

    @staticmethod
    def load_label_encoder(path: Optional[Path] = None) -> Any:
        """يحمّل مُرمِّز أسماء الأمراض (``sklearn.LabelEncoder``). يرفع
        ``ModelLoadError`` عند غياب الملف أو تعذّر التحميل أو غياب ``classes_``."""
        resolved_path = path or _DEFAULT_LABEL_ENCODER_PATH
        if not resolved_path.exists():
            raise ModelLoadError(f"ملف label_encoder غير موجود: {resolved_path}")

        try:
            encoder = joblib.load(resolved_path)
        except Exception as exc:  # noqa: BLE001 - أي فشل تحميل يُغلَّف بنوع خطأ الدومين
            raise ModelLoadError(f"تعذّر تحميل label_encoder ({resolved_path}): {exc}") from exc

        if not hasattr(encoder, "classes_"):
            raise ModelLoadError(
                f"الكائن المحمَّل من {resolved_path} لا يوفّر classes_ — ليس مُرمِّز فئات صالحاً."
            )
        return encoder

    @staticmethod
    def load_all(
        model_path: Optional[Path] = None,
        feature_names_path: Optional[Path] = None,
        label_encoder_path: Optional[Path] = None,
    ) -> Tuple[Any, List[str], Any]:
        """يحمّل الثلاثة معاً ويتحقّق من تطابقها المتبادل — التحقّق الذي لا
        يمكن لأيٍّ من المحمِّلات الثلاثة الفردية إجراءه بمفرده.

        يرفع ``ModelLoadError`` عند: عدم تطابق ``model.n_features_in_`` مع طول
        ``feature_names``، أو عدم مطابقة أيٍّ منهما للأبعاد المتوقَّعة
        (``EXPECTED_N_FEATURES``/``EXPECTED_N_CLASSES``)، أو عدم تطابق عدد
        فئات ``label_encoder`` مع المتوقَّع.
        """
        model = ModelLoader.load_model(model_path)
        feature_names = ModelLoader.load_feature_names(feature_names_path)
        label_encoder = ModelLoader.load_label_encoder(label_encoder_path)

        n_features_model = getattr(model, "n_features_in_", None)
        if n_features_model != len(feature_names):
            raise ModelLoadError(
                f"عدم تطابق: model.n_features_in_={n_features_model} "
                f"لا يساوي طول feature_names={len(feature_names)}."
            )
        if n_features_model != EXPECTED_N_FEATURES:
            raise ModelLoadError(
                f"عدد ميزات النموذج {n_features_model} لا يطابق المتوقَّع {EXPECTED_N_FEATURES}."
            )

        n_classes = len(getattr(label_encoder, "classes_", []))
        if n_classes != EXPECTED_N_CLASSES:
            raise ModelLoadError(
                f"عدد فئات label_encoder {n_classes} لا يطابق المتوقَّع {EXPECTED_N_CLASSES}."
            )

        return model, feature_names, label_encoder
