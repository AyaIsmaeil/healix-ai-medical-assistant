import os
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DISEASES_JSON_PATH = os.path.join(PROJECT_ROOT, "app", "database", "diseases.json")

# قاموس الترجمة الكامل لـ 49 عرضاً طبياً
SYMPTOM_TRANSLATION = {
    "abdominal_pain": "ألم بطن", "vomiting": "قيء", "nausea": "غثيان", "diarrhea": "إسهال",
    "constipation": "إمساك", "heartburn": "حرقة معدة", "loss_of_appetite": "فقدان شهية",
    "indigestion": "عسر هضم", "bloating": "نفخة وبطن منفوخ", "difficulty_swallowing": "صعوبة بلع",
    "cough": "سعال", "fever": "حرارة", "shortness_of_breath": "ضيق تنفس", "runny_nose": "رشح",
    "sore_throat": "وجع حلق", "sneezing": "عطاس", "chest_pain": "ألم صدر", "wheezing": "صوت صفير بالصدر",
    "nasal_congestion": "احتقان أنف", "headache": "صداع", "dizziness": "دوار", "fatigue": "تعب", 
    "insomnia": "أرق وصعوبة نوم", "anxiety": "قلق وتوتر", "depression": "اكتئاب وضيق خلق", 
    "blurred_vision": "غبيش بالعيون", "fainting": "غميان أو دقة غشية", "confusion": "تشوش وضياع", 
    "numbness": "تنميل أو خدران", "muscle_pain": "ألم عضلات", "joint_pain": "وجع mفاصل", 
    "back_pain": "وجع ظهر", "skin_rash": "طفح جلدي", "itching": "حكة بالجلد", "swelling": "تورم أو نفخة بالجسم",
    "painful_urination": "حرقة بالبول", "frequent_urination": "كثرة تبول", "blood_in_urine": "دم بالبول",
    "weight_loss": "نقصان وزن مفاجئ", "weight_gain": "زيادة وزن غير طبيعية", "sweating": "تعرق",
    "chills": "قشعريرة وبردية", "palpitations": "خفقان وسرعة بضربات القلب", "bleeding": "نزيف",
    "high_blood_pressure": "ارتفاع ضغط", "low_blood_pressure": "هبوط ضغط", "hair_loss": "تساقط شعر",
    "ear_pain": "وجع أذن", "toothache": "وجع أسنان"
}

# قاموس صياغة الأسئلة بالعامية السورية الموجهة للمريض
SYMPTOM_QUESTIONS = {
    "ألم بطن": "عم تحس بأي وجع أو مغص ببطنك؟", "قيء": "عم تستفرغ أو تراجع شي؟",
    "غثيان": "حاسس بـ لعيان نفس أو كابة معدة؟", "إسهال": "في عندك إسهال كمان؟",
    "إمساك": "عم تعاني من إمساك وتأخر بالخروج؟", "حرقة معدة": "في حرقة أو حموضة عم تطلع من معدتك لصدقك؟",
    "فقدان شهية": "حاسس شهيتك مسدودة وما الك نفس عالاكل بنوب؟", "عسر هضم": "عم تحس بتخمة أو الأكل واقف عالمعدمة وما عم ينهضم؟",
    "نفخة وبطن منفوخ": "بطنك منفوخ وفي غازات عم تزعجك؟", "صعوبة بلع": "عم تواجه صعوبة أو وجع وأنت عم تبلع الأكل؟",
    "سعال": "في عندك سعلة؟ ناشفة ولا معها بلغم؟", "حرارة": "حاسس بجسمك دافي أو في ارتفاع بالحرارة وسخونة؟",
    "ضيق تنفس": "عم تحس بضيق نفس أو كتمة صدر وأنت عم تتنفس؟", "رشح": "في عندك رشح أو سيلان بأنفك؟",
    "وجع حلق": "حلقك عم يوجعك أو في شحورة وصعوبة حكي؟", "عطاس": "عم تعاني من عطاس متكرر هالأيام؟",
    "ألم صدر": "في أي نغزة أو وجع وثقل بمنطقة الصدر？", "صوت صفير بالصدر": "عم تسمع صوت صفير أو هرير بصدرك وأنت عم تتنفس؟",
    "احتقان أنف": "أنفك مسدود وحاسس بضغط بجيوبك الأنفية؟", "صداع": "عم يشكي راسك من وجع أو شقيقة؟",
    "دوار": "عم تحس بدوخة والدني عم تدور فيك أو خفة براسك؟", "تعب": "حاسس بتعب، خمول، وهدّان حيل عام بجسمك؟",
    "أرق وصعوبة نوم": "عم تلاقي صعوبة بالنوم أو قلق بالليل؟", "قلق وتوتر": "حاسس بحالة قلق وتوتر زائد هالفترة؟",
    "اكتئاب وضيق خلق": "خلقك ضيق وحاسس بملل أو نكد ماله سبب؟", "غبيش بالعيون": "في أي غبش أو عدم وضوح بنظرك؟",
    "غميان أو دقة غشية": "هل غبت عن الوعي أو حسيت بدك يغمى عليك؟", "تشوش وضياع": "عم تحس بقلة تركيز أو نسيان وتشوش بالبال؟",
    "تنميل أو خدران": "في أي إحساس بالتنميل أو الخدر بأطرافك أو بوجهك؟", "ألم عضلات": "في وجع بعضلاتك أو حاسس جسمك مكسر تكسير؟",
    "وجع مفاصل": "مفاصلك عم توجعك أو فيهم قساوة بالحركة؟", "وجع ظهر": "عم تعاني من وجع بضهرك (بالأعلى أو بالأسفل)؟",
    "طفح جلدي": "ظهر أي حبوب، بقع حمراء أو طفح على جلدك؟", "حكة بالجلد": "في حكة قوية بجلدك أو عم تحس بتهييج؟",
    "تورم أو نفخة بالجسم": "لاحظت ورم أو نفخة بأيديك أو رجليك؟", "حرقة بالبول": "عم تحس بحرقة أو وجع وأنت عم تتبول؟",
    "كثرة تبول": "عم تروح عالحمام كتير وفوق العادة؟", "دم بالبول": "لاحظت أي تغير بلون البول أو نزول دم معه؟",
    "نقصان وزن مفاجئ": "نزل وزنك بالفترة الأخيرة بشكل سريع وبدون ريجيم؟", "زيادة وزن غير طبيعية": "حسيت بزيادة وزن مفاجئة ومنفوخة؟",
    "تعرق": "عم تعرق كتير، خصوصي بالليل وأنت نايم؟", "قشعريرة وبردية": "عم تجيك برديات مفاجئة وتنفض من البرد؟",
    "خفقان وسرعة بضربات القلب": "حاسس بقلبك عم يدق بسرعة أو ركض بصدرك؟", "نزيف": "صار معك نزيف من الأنف أو أي مكان تاني؟",
    "ارتفاع ضغط": "قست ضغطك وطلع عالي شي؟", "هبوط ضغط": "عم تحس بهبوط وضغطك واطي؟",
    "تساقط شعر": "شعر راسك عم يهر أو يتساقط بشكل ملحوظ؟", "وجع أذن": "في وجع أو تذكير وطنين بأدنيك؟",
    "وجع أسنان": "عم تتركب من وجع بأسنانك أو لثتك؟"
}

def get_diagnostic_questions(detected_symptoms_en):
    """
    تطابق الأعراض مع الـ JSON الخاص بك وترجع الأمراض والأسئلة القادمة
    """
    # تحويل الانجليزي لعربي
    detected_symptoms_ar = [SYMPTOM_TRANSLATION.get(sym, sym) for sym in detected_symptoms_en]
    detected_set = set(detected_symptoms_ar)

    try:
        with open(DISEASES_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ خطأ في تحميل الـ JSON الطبية: {e}")
        return {"possible_diseases": [], "next_questions": []}

    possible_diseases = []
    optional_symptoms_to_ask = set()

    for disease in data.get("diseases", []):
        req_symptoms = set(disease.get("symptoms_required", []))
        opt_symptoms = set(disease.get("symptoms_optional", []))
        all_disease_symptoms = req_symptoms.union(opt_symptoms)
        
        intersection = detected_set.intersection(all_disease_symptoms)
        if intersection:
            match_score = (len(intersection) / len(all_disease_symptoms)) * 100
            possible_diseases.append({
                "name": disease["name"],
                "risk": disease["risk_level"],
                "specialty": disease["specialty"],
                "match_score": f"{match_score:.1f}%"
            })
            # تجميع الأعراض الاختيارية المفقودة
            missing_optionals = opt_symptoms.difference(detected_set)
            optional_symptoms_to_ask.update(missing_optionals)

    # توليد الأسئلة السورية بناءً على الأعراض المفقودة
    next_questions = [SYMPTOM_QUESTIONS[sym] for sym in optional_symptoms_to_ask if sym in SYMPTOM_QUESTIONS]
    
    # ترتيب الأمراض من الأعلى احتمالية للأقل
    possible_diseases = sorted(possible_diseases, key=lambda x: x['match_score'], reverse=True)

    return {
        "possible_diseases": possible_diseases,
        "next_questions": next_questions[:2]  # نرجع سؤالين في الجولة الواحدة منعاً للتشتيت
    }