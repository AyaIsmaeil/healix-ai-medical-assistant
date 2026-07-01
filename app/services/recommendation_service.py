# service/recommendation_service.py
import json
import os

class RecommendationService:
    def __init__(self):
        self.specialty_map = {
            'قلبية': ['ألم صدر', 'ضيق تنفس', 'خفقان'],
            'أعصاب': ['صداع', 'دوار', 'تنميل', 'شلل'],
            'جهاز هضمي': ['غثيان', 'قيء', 'إسهال', 'ألم بطن'],
            'صدرية': ['سعال', 'ضيق تنفس', 'ربو'],
            'طب عام': ['حرارة', 'تعب', 'أعراض بسيطة'],
            'طوارئ': ['فقدان وعي', 'نزيف', 'ألم صدر شديد']
        }
    
    def recommend(self, symptoms, predictions):
        """
        اقتراح التخصص المناسب بناءً على الأعراض والتشخيص
        
        Args:
            symptoms: قائمة الأعراض
            predictions: قائمة الأمراض المتوقعة
            
        Returns:
            قاموس يحتوي على التخصص الموصى به
        """
        # إذا كان فيه توقع أول، استخدم تخصصه
        if predictions and len(predictions) > 0:
            top_disease = predictions[0]
            if 'specialty' in top_disease and top_disease['specialty']:
                return {
                    'specialty': top_disease['specialty'],
                    'specialty_en': self.get_specialty_en(top_disease['specialty']),
                    'reason': f'بناءً على التشخيص المحتمل: {top_disease["disease"]}',
                    'urgency': top_disease.get('risk_level', 'Medium')
                }
        
        # إذا ما في توقع، استخدم الأعراض لتحديد التخصص
        specialty_scores = {}
        for specialty, symptom_list in self.specialty_map.items():
            score = sum(1 for s in symptoms if s in symptom_list)
            if score > 0:
                specialty_scores[specialty] = score
        
        if specialty_scores:
            best_specialty = max(specialty_scores, key=specialty_scores.get)
            return {
                'specialty': best_specialty,
                'specialty_en': self.get_specialty_en(best_specialty),
                'reason': f'بناءً على الأعراض التي ذكرتها',
                'urgency': 'Medium'
            }
        
        # الإفتراضي
        return {
            'specialty': 'طبيب عام',
            'specialty_en': 'General Practitioner',
            'reason': 'للتقييم الأولي',
            'urgency': 'Low'
        }
    
    def get_specialty_en(self, specialty_ar):
        """تحويل اسم التخصص إلى الإنجليزية"""
        mapping = {
            'قلبية': 'Cardiology',
            'أعصاب': 'Neurology',
            'جهاز هضمي': 'Gastroenterology',
            'صدرية': 'Pulmonology',
            'طب عام': 'General Medicine',
            'طوارئ': 'Emergency Medicine',
            'طب باطني': 'Internal Medicine',
            'طبيب عام': 'General Practitioner'
        }
        return mapping.get(specialty_ar, 'General Medicine')


# اختبار الملف
if __name__ == "__main__":
    print("=" * 50)
    print("اختبار RecommendationService")
    print("=" * 50)
    
    recommender = RecommendationService()
    
    test_cases = [
        (['حرارة', 'سعال'], [{'disease': 'إنفلونزا', 'specialty': 'طب باطني'}]),
        (['ألم صدر', 'ضيق تنفس'], [{'disease': 'مرض قلبي', 'specialty': 'قلبية'}]),
        (['صداع'], [])
    ]
    
    for symptoms, predictions in test_cases:
        print(f"\n📋 الأعراض: {symptoms}")
        result = recommender.recommend(symptoms, predictions)
        print(f"   التخصص: {result['specialty']}")
        print(f"   السبب: {result['reason']}")