import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import json
import os

def main():
    print("=" * 60)
    print("🚀 بدء Fine-Tuning MARBERT لمشروع Healix")
    print("=" * 60)
    
    # -----------------------------
    # 1. تحميل البيانات
    # -----------------------------
    df = pd.read_csv('app/data/training_data_syrian_full.csv')
    print(f" إجمالي الجمل: {len(df)}")
    print(f"\n أول 5 صفوف:")
    print(df.head())
    
    # قائمة الأعراض الفريدة
    all_symptoms = []
    for symptoms_str in df['symptoms']:
        symptoms = symptoms_str.split('|')
        all_symptoms.extend(symptoms)
    
    unique_symptoms = sorted(list(set(all_symptoms)))
    print(f"\n عدد الأعراض الفريدة: {len(unique_symptoms)}")
    print(f"   الأعراض: {unique_symptoms}")
    
    # تحويل الأعراض إلى Multi-Label Binarizer
    mlb = MultiLabelBinarizer(classes=unique_symptoms)
    df['symptoms_list'] = df['symptoms'].apply(lambda x: x.split('|'))
    y = mlb.fit_transform(df['symptoms_list'])
    
    # -----------------------------
    # 2. تقسيم البيانات
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].tolist(),
        y,
        test_size=0.2,
        random_state=42,
        stratify=df['severity']
    )
    
    print(f"\n تدريب: {len(X_train)} جملة")
    print(f" تحقق: {len(X_val)} جملة")
    
    # -----------------------------
    # 3. تحميل MARBERT
    # -----------------------------
    model_name = "UBC-NLP/MARBERT"
    print(f"\n تحميل النموذج: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_symptoms),
        ignore_mismatched_sizes=True,
        problem_type="multi_label_classification"
    )
    
    # -----------------------------
    # 4. ترميز النصوص
    # -----------------------------
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    
    train_dataset = Dataset.from_dict({
        'text': X_train,
        'labels': y_train.tolist()
    })
    val_dataset = Dataset.from_dict({
        'text': X_val,
        'labels': y_val.tolist()
    })
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # -----------------------------
    # 5. دالة حساب المقاييس
    # -----------------------------
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        predictions = (predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(labels, predictions)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }
    
    # -----------------------------
    # 6. إعدادات التدريب
    # -----------------------------
    training_args = TrainingArguments(
        output_dir='./results/marbert_healix',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # smaller batch for smaller dataset
        per_device_eval_batch_size=8,
        num_train_epochs=10,  # more epochs for small dataset
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_total_limit=2,
        report_to='none'
    )
    
    # -----------------------------
    # 7. بدء التدريب
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("\n بدء Fine-Tuning...")
    print("   هذا قد يستغرق 10-20 دقيقة حسب جهازك")
    trainer.train()
    
    # -----------------------------
    # 8. حفظ النموذج
    # -----------------------------
    # os.makedirs('app/models/healix-marbert', exist_ok=True)
    model.save_pretrained('app/models/marbert')
    tokenizer.save_pretrained('app/models/marbert')
    
    # حفظ الـ MultiLabelBinarizer
    import joblib
    joblib.dump(mlb, 'app/models/marbert/mlb.pkl')
    
    # حفظ قائمة الأعراض
    with open('app/models/marbert/symptoms_list.json', 'w', encoding='utf-8') as f:
        json.dump(unique_symptoms, f, ensure_ascii=False, indent=2)
    
    print("\n تم حفظ النموذج في: app/models/marbert")
    
    # -----------------------------
    # 9. تقييم نهائي
    # -----------------------------
    eval_results = trainer.evaluate()
    print(f"\n نتائج التقييم النهائي:")
    print(f"   الدقة (Accuracy): {eval_results['eval_accuracy']:.2%}")
    print(f"   F1 Micro: {eval_results['eval_f1_micro']:.4f}")
    print(f"   F1 Macro: {eval_results['eval_f1_macro']:.4f}")

if __name__ == "__main__":
    main()