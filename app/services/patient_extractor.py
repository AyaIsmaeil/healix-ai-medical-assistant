"""
Enhanced Post-Processing Module
استخراج معلومات شاملة عن المريض
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


AGE_PATTERNS = [
    # Direct age mention
    r'عمري\s*(\d+)',
    r'أنا عمري\s*(\d+)',
    r'عندي\s*(\d+)\s*سنة',
    r'(\d+)\s*سنة',
    r'عمر\s*(\d+)',
    r'أنا بـ\s*(\d+)',
    
    # Age groups
    r'(عجوز|كبير بالسن)',
    r'(شاب|صغير)',
    r'(طفل|ولد|بنت)',
    r'(رضيع|طفل صغير)',
    r'(مراهق|فتى)',
]

def extract_age(text: str) -> Optional[Dict]:
    """استخرج العمر من النص"""
    
    # محاولة استخراج رقم العمر
    for pattern in AGE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                # محاولة الحصول على الرقم
                age_str = match.group(1)
                if age_str.isdigit():
                    age = int(age_str)
                    
                    # تصنيف العمر
                    if age < 2:
                        age_group = "infant"
                    elif age < 13:
                        age_group = "child"
                    elif age < 18:
                        age_group = "teenager"
                    elif age < 65:
                        age_group = "adult"
                    else:
                        age_group = "elderly"
                    
                    return {
                        'age': age,
                        'age_group': age_group,
                        'source': 'explicit'
                    }
            except:
                pass
    
    return None

CHRONIC_DISEASE_PATTERNS = {
    'diabetes': [
        r'سكري|diabetes|سكر|سكيري',
        r'النوع الأول|النوع الثاني',
    ],
    'hypertension': [
        r'ضغط|ارتفاع ضغط|pressure|ضغط الدم',
    ],
    'heart_disease': [
        r'قلب|مرض قلب|heart|أمراض القلب',
        r'جلطة|stroke',
    ],
    'asthma': [
        r'ربو|asthma',
    ],
    'kidney_disease': [
        r'كلى|kidney|مرض كلى',
    ],
    'thyroid': [
        r'غدة درقية|thyroid',
    ],
    'cancer': [
        r'سرطان|cancer|ورم',
    ],
    'arthritis': [
        r'روماتيزم|arthritis|التهاب المفاصل',
    ],
    'liver_disease': [
        r'كبد|liver|مرض كبد',
    ],
    'copd': [
        r'انسداد الشعب|COPD|emphysema',
    ],
    'depression': [
        r'اكتئاب|depression|حالة نفسية',
    ],
    'anxiety': [
        r'قلق|anxiety|توتر',
    ],
}

def extract_chronic_diseases(text: str) -> List[str]:
    """استخرج الأمراض المزمنة"""
    
    text_lower = text.lower()
    detected_diseases = set()
    
    for disease, patterns in CHRONIC_DISEASE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                detected_diseases.add(disease)
                break  # لا نحتاج للبحث أكتر عن هاد المرض
    
    return sorted(list(detected_diseases))


MEDICATION_PATTERNS = {
    'aspirin': [r'أسبرين|aspirin'],
    'ibuprofen': [r'ايبوبروفين|ibuprofen'],
    'paracetamol': [r'بارسيتامول|paracetamol|تايلينول'],
    'antibiotics': [r'مضاد حيوي|antibiotic'],
    'antacid': [r'مضاد حموضة|antacid'],
    'antihistamine': [r'مضاد حساسية|antihistamine'],
    'diuretic': [r'مدر بول|diuretic'],
    'beta_blocker': [r'بيتا بلوكر|beta blocker'],
    'ace_inhibitor': [r'مثبط ACE|ACE inhibitor'],
    'statin': [r'ستاتين|statin'],
    'insulin': [r'أنسولين|insulin'],
    'corticosteroid': [r'كورتيكوستيرويد|corticosteroid|كورتيزون'],
}

def extract_medications(text: str) -> List[Dict]:
    """استخرج الأدوية المستخدمة"""
    
    text_lower = text.lower()
    detected_meds = []
    
    for med, patterns in MEDICATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                detected_meds.append({
                    'medication': med,
                    'mentioned': True
                })
                break
    
    # استخرج الأدوية المذكورة بشكل مباشر
    med_mentions = re.findall(r'(شاربة|آخذ|أستخدم|بتناول)\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)', text)
    for _, med_name in med_mentions:
        med_name = med_name.strip()
        if med_name and len(med_name) > 2:
            detected_meds.append({
                'medication': med_name,
                'mentioned': True
            })
    
    # أيضاً: "ما بحب الأسبرين"
    negative_patterns = r'(?:ما بحب|بتحس برد فعل|حساس من|ما قدر)\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)'
    allergies = re.findall(negative_patterns, text)
    for allergy in allergies:
        detected_meds.append({
            'medication': allergy.strip(),
            'allergy': True
        })
    
    return detected_meds


ALLERGY_PATTERNS = [
    r'(?:حساس|ما بقدر|أنا حساس|بتحس برد فعل)\s+(?:من|ل)\s*([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    r'حساسية\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    r'أرجيا\s+(?:من|ل)\s*([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    r'([أ-ي\s]+?)\s+(?:حساسية|حساس|برد فعل)',
]

def extract_allergies(text: str) -> List[Dict]:
    """استخرج الحساسيات"""
    
    detected_allergies = []
    
    # Common allergies
    common_allergies = {
        'peanuts': ['فول سوداني', 'peanuts'],
        'shellfish': ['جمبري', 'أسماك', 'shellfish'],
        'dairy': ['حليب', 'جبن', 'dairy', 'lactose'],
        'gluten': ['جلوتين', 'gluten', 'قمح'],
        'eggs': ['بيض', 'eggs'],
        'tree_nuts': ['جوز', 'بندق', 'nuts'],
        'sulfites': ['كبريتات', 'sulfites'],
        'sesame': ['سمسم', 'sesame'],
    }
    
    text_lower = text.lower()
    
    for allergy_type, keywords in common_allergies.items():
        for keyword in keywords:
            if re.search(keyword, text_lower):
                detected_allergies.append({
                    'type': allergy_type,
                    'allergy': keyword
                })
                break
    
    # استخرج من الأنماط
    for pattern in ALLERGY_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            allergy_name = match.strip()
            if allergy_name and len(allergy_name) > 2:
                detected_allergies.append({
                    'type': 'other',
                    'allergy': allergy_name
                })
    
    return detected_allergies

#
FAMILY_HISTORY_PATTERNS = [
    r'(?:أمي|والديي|أبي|جدي|جدتي|أختي|أخي|عمي|خالتي)\s+(?:عندها|عنده)\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    r'(?:في عايلتنا|في البيت)\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    r'تاريخ عائلي\s+من\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
]

def extract_family_history(text: str) -> List[Dict]:
    """استخرج السجل العائلي"""
    
    family_members = ['أم', 'أب', 'أخ', 'أخت', 'عم', 'عمة', 'خال', 'خالة', 'جد', 'جدة']
    diseases = ['ضغط', 'سكري', 'قلب', 'سرطان', 'ربو', 'اكتئاب']
    
    detected_family_history = []
    text_lower = text.lower()
    
    for member in family_members:
        for disease in diseases:
            pattern = f'{member}.*{disease}|{disease}.*{member}'
            if re.search(pattern, text_lower):
                detected_family_history.append({
                    'relation': member,
                    'condition': disease
                })
    
    # استخرج من الأنماط
    for pattern in FAMILY_HISTORY_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            detected_family_history.append({
                'relation': 'family',
                'condition': match.strip()
            })
    
    return detected_family_history

#
RECENT_EVENT_PATTERNS = {
    'recent_travel': [
        r'(سفرت|سفرنا|رجعنا من|كنا بـ|كانت)\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    ],
    'recent_infection': [
        r'(?:بعد ما|بعد|أثناء)\s+(نزلة|فيروس|عدوى|إنفلونزا)',
    ],
    'recent_stress': [
        r'(ضغط نفسي|توتر|قلق|خوف|حادثة)\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    ],
    'recent_surgery': [
        r'(عملية جراحية|جراحة|عملية)\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    ],
    'recent_accident': [
        r'(حادثة|اصطدام|سقطت|وقعت)\s+([أ-ي\s]+?)(?:\s+و|\s+،|\.)',
    ],
}

def extract_recent_events(text: str) -> List[Dict]:
    """استخرج الأحداث الحديثة"""
    
    detected_events = []
    text_lower = text.lower()
    
    for event_type, patterns in RECENT_EVENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                detected_events.append({
                    'type': event_type,
                    'description': match.group(0).strip()
                })
    
    return detected_events


LIFESTYLE_PATTERNS = {
    'smoker': [r'دخان|سيجارة|smoking|cigarette'],
    'alcohol': [r'خمر|alcohol|مشروبات'],
    'exercise': [r'رياضة|تمرين|exercise|gym'],
    'diet': [r'نظام غذائي|diet|vegetarian|веган'],
    'stress': [r'ضغط|توتر|stress|anxiety'],
    'sleep_issue': [r'نوم|sleep|أرق|insomnia'],
}

def extract_lifestyle(text: str) -> Dict[str, bool]:
    """استخرج معلومات نمط الحياة"""
    
    text_lower = text.lower()
    lifestyle = {}
    
    for lifestyle_factor, keywords in LIFESTYLE_PATTERNS.items():
        for keyword in keywords:
            if re.search(keyword, text_lower):
                lifestyle[lifestyle_factor] = True
                break
        else:
            lifestyle[lifestyle_factor] = False
    
    return lifestyle

@dataclass
class PatientInfo:
    """معلومات المريض الكاملة"""
    # الأساسية
    text: str
    
    # الأعراض (من MARBERT)
    symptoms: List[str]
    confidence: Dict[str, float]
    
    # المعلومات من POST-PROCESSING
    age: Optional[Dict] = None
    chronic_diseases: List[str] = None
    medications: List[Dict] = None
    allergies: List[Dict] = None
    family_history: List[Dict] = None
    recent_events: List[Dict] = None
    lifestyle: Dict[str, bool] = None
    
    # المعلومات الزمنية والحدة
    duration: Optional[str] = None
    severity: Optional[str] = None
    onset: Optional[str] = None
    red_flags: bool = False
    affected_areas: List[str] = None
    triggers: List[str] = None

def comprehensive_patient_extraction(
    text: str,
    extracted_symptoms: List[str],
    model_confidence: Dict[str, float],
    post_processing_result: Dict = None
) -> PatientInfo:
    """
    معالجة شاملة لاستخراج كل معلومات المريض
    """
    
    # معالجة منفصلة للمعلومات الجديدة
    age = extract_age(text)
    chronic_diseases = extract_chronic_diseases(text)
    medications = extract_medications(text)
    allergies = extract_allergies(text)
    family_history = extract_family_history(text)
    recent_events = extract_recent_events(text)
    lifestyle = extract_lifestyle(text)
    
    # المعلومات من post_processing_result (إن وجد)
    if post_processing_result is None:
        post_processing_result = {}
    
    return PatientInfo(
        text=text,
        symptoms=extracted_symptoms,
        confidence=model_confidence,
        age=age,
        chronic_diseases=chronic_diseases,
        medications=medications,
        allergies=allergies,
        family_history=family_history,
        recent_events=recent_events,
        lifestyle=lifestyle,
        duration=post_processing_result.get('duration'),
        severity=post_processing_result.get('severity'),
        onset=post_processing_result.get('onset'),
        red_flags=post_processing_result.get('red_flags', False),
        affected_areas=post_processing_result.get('affected_areas', []),
        triggers=post_processing_result.get('triggers', [])
    )


if __name__ == "__main__":
    print("="*80)
    print(" COMPREHENSIVE PATIENT INFORMATION EXTRACTION")
    print("="*80)
    
    # رسالة من المريض مع معلومات إضافية
    patient_message = """
    من امبارح معدتي عم توجعني وأنا عمري 45 سنة وعندي ضغط 
    وما بحب الأسبرين والتايلينول، بس بستخدم المتفورمين للسكري.
    وأمي عندها نفس مشاكل المعدة، وأخي عنده قلب.
    شغلي إجهاد كتير وما بنام منيح من فترة.
    """
    
    # نتائج من الموديل
    model_output = {
        'abdominal_pain': 0.92,
        'fever': 0.44,  # منخفضة
        'nausea': 0.65
    }
    
    # النتائج من post_processing السابق
    post_processing_result = {
        'duration': '1 day',
        'severity': 'moderate',
        'onset': 'sudden',
        'red_flags': False,
        'affected_areas': ['abdomen'],
        'triggers': []
    }
    
    # الاستخراج الشامل
    patient_info = comprehensive_patient_extraction(
        text=patient_message,
        extracted_symptoms=list(model_output.keys()),
        model_confidence=model_output,
        post_processing_result=post_processing_result
    )
    
    # اعرض النتائج
    print("\n EXTRACTED PATIENT INFORMATION:")
    print("-" * 80)
    
    print(f"\n العمر:")
    if patient_info.age:
        print(f"   العمر: {patient_info.age['age']} سنة ({patient_info.age['age_group']})")
    
    print(f"\n الأعراض الحالية:")
    for symptom, conf in zip(patient_info.symptoms, 
                             [model_output.get(s, 0) for s in patient_info.symptoms]):
        print(f"   • {symptom}: {conf:.0%}")
    
    print(f"\n الأمراض المزمنة:")
    if patient_info.chronic_diseases:
        for disease in patient_info.chronic_diseases:
            print(f"   • {disease}")
    else:
        print("   لا توجد أمراض مزمنة مذكورة")
    
    print(f"\n الأدوية المستخدمة:")
    if patient_info.medications:
        for med in patient_info.medications:
            status = "حساسية" if med.get('allergy') else "يستخدم"
            print(f"   • {med.get('medication')}: {status}")
    else:
        print("   لا توجد أدوية مذكورة")
    
    print(f"\n🔹 الحساسيات:")
    if patient_info.allergies:
        for allergy in patient_info.allergies:
            print(f"   • {allergy.get('allergy')} ({allergy.get('type')})")
    else:
        print("   لا توجد حساسيات مذكورة")
    
    print(f"\n السجل العائلي:")
    if patient_info.family_history:
        for fh in patient_info.family_history:
            print(f"   • {fh.get('relation')}: {fh.get('condition')}")
    else:
        print("   لا يوجد سجل عائلي مذكور")
    
    print(f"\n الأحداث الحديثة:")
    if patient_info.recent_events:
        for event in patient_info.recent_events:
            print(f"   • {event.get('type')}: {event.get('description')}")
    else:
        print("   لا توجد أحداث حديثة مذكورة")
    
    print(f"\n نمط الحياة:")
    for factor, present in patient_info.lifestyle.items():
        status = "✅" if present else "❌"
        print(f"   {status} {factor}")
    
    print(f"\n المعلومات الزمنية:")
    print(f"   المدة: {patient_info.duration}")
    print(f"   الشدة: {patient_info.severity}")
    print(f"   البداية: {patient_info.onset}")
    print(f"   علامات خطرة: {'نعم' if patient_info.red_flags else 'لا'}")
    
    print("\n" + "="*80)
    print(" استخراج كامل وشامل لمعلومات المريض")
    print("="*80)