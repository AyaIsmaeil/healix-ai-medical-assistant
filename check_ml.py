"""فحص يدوي سريع لـMLDiseasePredictor عبر النموذج الفعلي المحفوظ في models/
(لا نموذج اصطناعي كما في الاختبارات). يبني EncodedFeatures بثلاثة رموز E_*
حقيقية (أول ثلاثة في feature_names.json) + AGE + SEX_M، ويطبع أعلى الأمراض
المتوقَّعة مع احتمالاتها."""

from app.domain.feature_encoder import EncodedFeatures
from app.domain.ml_disease_predictor import MLDiseasePredictor
from app.infrastructure.model_loader import ModelLoader


def main() -> None:
    model, feature_names, label_encoder = ModelLoader.load_all()
    print(f"النموذج: {len(feature_names)} ميزة، {len(label_encoder.classes_)} فئة")

    evidence_codes = [name for name in feature_names if name.startswith("E_")][:3]
    print(f"الرموز المُختارة (أول 3 من feature_names.json): {evidence_codes}")

    features = {code: 1 for code in evidence_codes}
    features["age"] = 35
    features["gender_male"] = 1

    encoded = EncodedFeatures(
        feature_schema_version="check-ml-manual",
        features=features,
        categorical_index={},
    )

    predictor = MLDiseasePredictor(model=model, feature_names=feature_names, label_encoder=label_encoder)
    result = predictor.predict(encoded)

    print(f"\nإصدار المُتنبِّئ: {result.predictor_version}")
    print("الأمراض المتوقَّعة:")
    for prediction in result.predictions:
        print(f"  {prediction.disease:45s} {prediction.score * 100:5.1f}%  — {prediction.explanation}")


if __name__ == "__main__":
    main()
