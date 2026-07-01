# service/triage_service.py
class TriageService:
    def __init__(self):
        # أعراض عالية الخطورة
        self.high_risk_symptoms = [
            'ألم صدر', 'chest_pain',
            'ضيق تنفس', 'shortness_of_breath',
            'صعوبة تنفس', 'difficulty_breathing'
        ]
        
        # أعراض خطيرة جداً (طارئة)
        self.emergency_symptoms = [
            'فقدان وعي', 'loss_of_consciousness',
            'نزيف', 'bleeding',
            'شلل', 'paralysis',
            'حساسية مفرطة', 'severe_allergy'
        ]
    
    def classify(self, symptoms, age=30, chronic_diseases=None):
        """
        تصنيف خطورة الحالة
        
        Args:
            symptoms: قائمة الأعراض
            age: عمر المريض
            chronic_diseases: الأمراض المزمنة (string أو list)
            
        Returns:
            قاموس يحتوي على التصنيف والإجراء الموصى به
        """
        score = 0
        
        # 1. درجة الأعراض
        symptom_score = 0
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in self.emergency_symptoms:
                symptom_score = 100
                break
            elif symptom_lower in self.high_risk_symptoms:
                symptom_score += 35
        
        # 2. درجة العمر
        age_score = 0
        if age >= 65:
            age_score = 25
        elif age >= 50:
            age_score = 15
        elif age <= 5:
            age_score = 20
        
        # 3. درجة الأمراض المزمنة
        chronic_score = 0
        if chronic_diseases:
            chronic_list = chronic_diseases if isinstance(chronic_diseases, list) else [chronic_diseases]
            chronic_map = {
                'diabetes': 15, 'سكري': 15,
                'hypertension': 20, 'ضغط': 20,
                'asthma': 15, 'ربو': 15,
                'heart_disease': 30, 'قلب': 30
            }
            for chronic in chronic_list:
                chronic_lower = chronic.lower()
                for key, value in chronic_map.items():
                    if key in chronic_lower:
                        chronic_score = max(chronic_score, value)
        
        total_score = min(symptom_score + age_score + chronic_score, 100)
        
        # تصنيف الخطورة
        if total_score >= 85 or symptom_score == 100:
            return {
                'triage': 'Emergency',
                'triage_ar': 'طارئة',
                'level': 4,
                'color': '🔴',
                'score': total_score,
                'action': 'اتصل بالإسعاف فوراً أو اذهب لأقرب مستشفى',
                'action_en': 'Call ambulance immediately or go to nearest ER',
                'response_time': '0 دقيقة'
            }
        elif total_score >= 60:
            return {
                'triage': 'High',
                'triage_ar': 'عالية',
                'level': 3,
                'color': '🟠',
                'score': total_score,
                'action': 'يجب مراجعة الطوارئ خلال ساعة',
                'action_en': 'Visit ER within 1 hour',
                'response_time': '1 ساعة'
            }
        elif total_score >= 30:
            return {
                'triage': 'Medium',
                'triage_ar': 'متوسطة',
                'level': 2,
                'color': '🟡',
                'score': total_score,
                'action': 'ينصح بمراجعة طبيب خلال 24 ساعة',
                'action_en': 'See a doctor within 24 hours',
                'response_time': '24 ساعة'
            }
        else:
            return {
                'triage': 'Low',
                'triage_ar': 'بسيطة',
                'level': 1,
                'color': '🟢',
                'score': total_score,
                'action': 'متابعة منزلية مع مراقبة الأعراض',
                'action_en': 'Home care with symptom monitoring',
                'response_time': '48 ساعة'
            }


# اختبار الملف
if __name__ == "__main__":
    print("=" * 50)
    print("اختبار TriageService")
    print("=" * 50)
    
    triage = TriageService()
    
    test_cases = [
        (['حرارة'], 30, None),
        (['ألم صدر', 'ضيق تنفس'], 55, 'hypertension'),
        (['صداع'], 25, None),
        (['فقدان وعي'], 40, None)
    ]
    
    for symptoms, age, chronic in test_cases:
        print(f"\n📋 الأعراض: {symptoms}, العمر: {age}")
        result = triage.classify(symptoms, age, chronic)
        print(f"   الخطورة: {result['triage_ar']} ({result['color']})")
        print(f"   الإجراء: {result['action']}")