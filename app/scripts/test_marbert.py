# app/scripts/test_marbert.py
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# 1. ضبط المسارات لضمان التعرف على مجلد app
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "app", "data", "raw", "symptoms_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "app", "models", "marbert_symptoms")

print("[Evaluation] جاري تحضير بيئة تقييم نموذج Healix...")

# 2. تحميل البيانات وتجهيز داتا الفحص (Test Data)
try:
    df = pd.read_csv(DATA_PATH)
    symptom_columns = [col for col in df.columns if col != 'text']
    
    # لغرض التقييم، سنأخذ آخر 20% من البيانات كعينة فحص (Test Set) لم يرها الموديل أثناء التدريب
    # (إذا كان لديك ملف منفصل للفحص test.csv قم بتعديل المسار لقرائته مباشرة)
    test_size = int(len(df) * 0.2)
    test_df = df.tail(test_size).copy()
    print(f" تم تحميل البيانات. حجم عينة الفحص: {len(test_df)} جملة عامية.")
except Exception as e:
    raise FileNotFoundError(f"فشل تحميل الداتا سيت لغرض الفحص. الخطأ: {e}")

# 3. تحميل الموديل والتوكنايزر محلياً
print(f"جاري تحميل الموديل محلياً من: {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 4. دالة التنبؤ الجماعي (Batch Prediction) لتسريع عملية الفحص
def evaluate_model(df, threshold=0.5):
    texts = df['text'].tolist()
    # تحويل الإجابات الحقيقية من الـ CSV إلى مصفوفة (Ground Truth)
    y_true = df[symptom_columns].values
    y_pred = []

    print(" جاري تشغيل التنبؤات على عينة الفحص وتمريرها عبر معالج الـ MARBERT...")
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            # تطبيق Sigmoid للحصول على الاحتمال لكل عرض بشكل مستقل (Multi-label)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            
            # إذا كان الاحتمال أكبر من أو يساوي الـ Threshold، نعتبر العرض موجوداً (1) وإلا (0)
            preds = (probs >= threshold).astype(int)
            y_pred.append(preds)
            
    return y_true, np.array(y_pred)

if __name__ == "__main__":
    # تشغيل الفحص عند عتبة ثقة 0.5
    THRESHOLD = 0.5
    y_true, y_pred = evaluate_model(test_df, threshold=THRESHOLD)
    
    # 5. حساب المقاييس الإحصائية الكلية (Global Metrics)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    
    print("\n" + "="*60)
    print(" نتيجـة تقييـم الأداء العـام للموديـل (Global Performance Report)")
    print("="*60)
    print(f" عتبة القرار المعتمدة (Confidence Threshold): {THRESHOLD}")
    print(f" دقة التحديد (Micro Precision): {precision:.4f} ({precision*100:.2f}%)")
    print(f" استدعاء الأعراض (Micro Recall):  {recall:.4f} ({recall*100:.2f}%)")
    print(f" مقياس إف-1 الشامل (Micro F1-Score): {micro_f1:.4f} ({micro_f1*100:.2f}%) -> (موصى به للموازنة)")
    print(f" مقياس إf-1 غير الموزون (Macro F1-Score): {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print("="*60)
    
    # 6. طباعة تقرير مفصل لكل عرض من الأعراض الـ 49 التي تم التنبؤ بها
    print("\n تقرير الأداء التفصيلي لكل عرض طبي (Detailed Classification Report):")
    # نستخدم zero_division=0 لتجنب المشاكل البرمجية للأعراض النادرة في عينة الفحص
    report = classification_report(y_true, y_pred, target_names=symptom_columns, zero_division=0)
    print(report)