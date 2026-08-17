# Healix — شرح مراحل النظام

> مساعد طبي ذكي باللغة العربية (لهجة سورية)، يعمل كخدمة Python يستهلكها backend Laravel عبر HTTP.
> هذا الملف يشرح **كل مرحلة (Node)** في الـ Graph، و**ما الذي تستخدمه** في كل مرحلة.

---

## نظرة عامة على التدفق

```
START
  → reset_stage
  → crisis_check ── أزمة ──► crisis_node ──► END
  → extract_symptoms
  → check_red_flags ── علامة خطر ──► emergency_node ──► END
                     ── نتيجة سابقة بدون تصعيد جديد ──► reiterate_terminal_outcome ──► END
  → assess_sufficiency
       ├── معلومات غير كافية ──► ask_followup ──► END
       └── معلومات كافية ──► rag_retrieve
                                  ├── يحتاج تأكيد الجنس ──► ask_followup ──► END
                                  └── otherwise ──► diagnose → route_specialty → generate_reports ──► END
```

**المراحل النهائية (Terminal Stages)** التي يراها المريض:

| `stage` | المعنى | Node |
|---------|--------|------|
| `crisis` | أزمة نفسية / إيذاء ذاتي | `crisis_node` |
| `emergency` | حالة طوارئ طبية | `emergency_node` |
| `followup` | سؤال متابعة | `ask_followup` |
| `diagnosis` | تشخيص تفاضلي + تقارير | `generate_reports` |

---

## البنية التحتية المشتركة

| المكوّن | الملف | الدور |
|---------|-------|-------|
| **Graph** | `graph.py` | يربط كل الـ Nodes ويحدد المسارات الشرطية |
| **State** | `state.py` | حالة المحادثة المشتركة (`HealixState`) |
| **LLM Client** | `llm_client.py` | نقطة واحدة لكل استدعاءات LLM (Gemini / Groq / Ollama) |
| **Prompts** | `prompts/base.py` + `prompts/templates/` | بناء الـ prompts مع safety preamble |
| **Rules** | `rules/` | كشف حتمي (deterministic) للأزمات وعلامات الخطر |
| **Schemas** | `schemas/` | نماذج Pydantic للمخرجات المنظّمة |
| **RAG KB** | `rag/knowledge_base/*.json` | قاعدة معرفة الأمراض (~40 مرض) |
| **Vocabulary** | `vocabulary/` | مفردات الأعراض المعيارية |
| **API** | `api/main.py` | FastAPI — `POST /chat` + واجهة dev |
| **Persistence** | LangGraph Checkpointer | SQLite (تطوير) أو PostgreSQL (إنتاج) |

### مستويات LLM (Tiers)

| Tier | الاستخدام |
|------|-----------|
| `quality` | كل Node يقرأ/يفسر رسالة المريض العربية — **إلزامي للأمان** |
| `fast` | أدوات bulk خارج الـ Graph (مثل `scripts/filter_columbia_symptoms.py`) |

---

## المراحل بالتفصيل

---

### 1. `reset_stage`

| | |
|---|---|
| **الملف** | `nodes/reset_stage.py` |
| **الهدف** | مسح `stage` من الدورة السابقة قبل بدء دورة جديدة |
| **LLM** | ❌ لا |
| **Rules** | ❌ |
| **Prompts** | ❌ |
| **Schemas** | ❌ |

**ملاحظة:** لا يمسح `thread_outcome` — النتيجة الأمنية السابقة (أزمة/طوارئ) تبقى ثابتة طوال المحادثة.

---

### 2. `crisis_check`

| | |
|---|---|
| **الملف** | `nodes/crisis_check.py` |
| **الهدف** | كشف أزمة نفسية أو إيذاء ذاتي — **أول Node بعد reset** |
| **LLM** | ✅ `quality` tier |
| **Prompt** | `prompts/templates/crisis_check.txt` |
| **Schema** | `schemas/crisis.py` → `CrisisCheckResult` |
| **Rules** | `rules/crisis.py` → `detect_crisis()` |
| **Audit** | `audit/logger.py` → `log_crisis_detection()` |

**آلية الكشف:** Rule-based **OR** LLM — أيّهما يكتشف = أزمة.

**مدخلات State:** `messages` (آخر رسالة مريض + السؤال السابق)
**مخرجات State:** `is_crisis: bool`

**التوجيه:** إذا `is_crisis=True` → `crisis_node` | وإلا → `extract_symptoms`

---

### 3. `crisis_node` ⚠️ Terminal

| | |
|---|---|
| **الملف** | `nodes/crisis_node.py` |
| **الهدف** | رسالة أزمة للمريض + خطوط دعم |
| **LLM** | ✅ `quality` tier |
| **Prompt** | `prompts/templates/crisis_node.txt` |
| **Schema** | `schemas/crisis.py` → `CrisisResponse` |
| **Rules** | ❌ |
| **Code** | `support_lines.py` — أرقام الخطوط الساخنة **تُدرَج بالكود** وليس من LLM |

**مخرجات State:**
- `stage = "crisis"`
- `thread_outcome = "crisis"`
- `messages` ← رد المساعد

**قاعدة أمان:** يتجاوز RAG والتشخيص بالكامل.

---

### 4. `extract_symptoms`

| | |
|---|---|
| **الملف** | `nodes/extract_symptoms.py` |
| **الهدف** | استخراج الأعراض المؤكدة والمنفية من رسالة المريض |
| **LLM** | ✅ `quality` tier |
| **Prompt** | `prompts/templates/extract_symptoms.txt` |
| **Schema** | `schemas/symptoms.py` → `SymptomExtraction` |
| **Vocabulary** | `vocabulary/symptoms.py`, `vocabulary/duration.py`, `vocabulary/severity.py` |
| **Rules** | ❌ |

**مدخلات إضافية للـ prompt:**
- `previous_question` — آخر سؤال من المساعد (لربط إجابات قصيرة مثل "من الصبح")
- `known_symptoms` — الأعراض المتراكمة من الدورات السابقة

**مخرجات State:**
- `symptoms` — تُدمَج عبر reducer `merge_symptoms`
- `negated_symptoms` — تُدمَج بنفس الطريقة
- `unmatched_mentions` — عبارات لم تُطابق المفردات

---

### 5. `check_red_flags`

| | |
|---|---|
| **الملف** | `nodes/check_red_flags.py` |
| **الهدف** | كشف علامات الخطر الطبية (طوارئ) |
| **LLM** | ✅ `quality` tier |
| **Prompt** | `prompts/templates/check_red_flags.txt` |
| **Schema** | `schemas/red_flags.py` → `RedFlagCheckResult` |
| **Rules** | `rules/red_flags.py` → `check_red_flags()` |
| **Negation** | `rules/negation.py` — للتعامل مع النفي في النص |

**آلية الكشف:** Rule-based **OR** LLM — أيّهما يكتشف = علامة خطر.

**مخرجات State:** `red_flags: list[{"id", "reason"}]`

**التوجيه:**
- إذا `red_flags` غير فارغ → `emergency_node`
- إذا `thread_outcome` موجود (أزمة/طوارئ سابقة) → `reiterate_terminal_outcome`
- وإلا → `assess_sufficiency`

---

### 6. `emergency_node` 🚨 Terminal

| | |
|---|---|
| **الملف** | `nodes/emergency_node.py` |
| **الهدف** | رسالة طوارئ ثابتة — توجيه فوري للطوارئ |
| **LLM** | ❌ لا — رسالة hand-authored ثابتة |
| **Rules** | ❌ |

**مخرجات State:**
- `stage = "emergency"`
- `thread_outcome = "emergency"`
- `severity = "emergency"`
- `messages` ← رسالة ثابتة بالعربية

**قاعدة أمان:** لا يشرح *لماذا* للمريض (تجنب تشخيص). التفاصيل في `red_flags` للطبيب.

---

### 7. `reiterate_terminal_outcome` ⚠️ Terminal

| | |
|---|---|
| **الملف** | `nodes/reiterate_terminal_outcome.py` |
| **الهدف** | تذكير المريض بالتوجيه السابق (أزمة/طوارئ) بدون إعادة triage |
| **LLM** | ❌ لا — رسالة ثابتة |
| **Rules** | ❌ |

**متى يُفعَّل:** محادثة وصلت سابقاً لـ crisis/emergency، والدورة الحالية لم تكتشف تصعيداً جديداً.

**مخرجات State:** `stage` = `"crisis"` أو `"emergency"` (حسب `thread_outcome`)

---

### 8. `assess_sufficiency`

| | |
|---|---|
| **الملف** | `nodes/assess_sufficiency.py` |
| **الهدف** | هل المعلومات كافية للتشخيص التفاضلي؟ |
| **LLM** | ✅ `quality` tier |
| **Prompt** | `prompts/templates/assess_sufficiency.txt` |
| **Schema** | `schemas/sufficiency.py` → `SufficiencyAssessment` |
| **Rules** | ❌ |

**ثابت:** `MAX_FOLLOW_UP_QUESTIONS = 3` — حد أقصى لأسئلة المتابعة.

**مخرجات State:**
- `is_sufficient: bool`
- `next_question: str | None` — سؤال المتابعة بالعربية السورية
- `information_limited: bool` — true إذا وصلنا للحد واضطررنا للمتابعة
- `turn_count` — يُزاد +1 عند طلب سؤال متابعة

**التوجيه:**
- `is_sufficient=False` → `ask_followup`
- `is_sufficient=True` → `rag_retrieve`

---

### 9. `ask_followup` 💬 Terminal

| | |
|---|---|
| **الملف** | `nodes/ask_followup.py` |
| **الهدف** | إرسال سؤال المتابعة للمريض وانتظار رده |
| **LLM** | ❌ لا — يقرأ `next_question` الجاهز |
| **Rules** | ❌ |

**مخرجات State:**
- `stage = "followup"`
- `messages` ← `next_question`

**يُستخدم من:** `assess_sufficiency` (سؤال عن الأعراض) و `rag_retrieve` (تأكيد الجنس).

---

### 10. `rag_retrieve`

| | |
|---|---|
| **الملف** | `nodes/rag_retrieve.py` |
| **الهدف** | مطابقة الأعراض مع قاعدة المعرفة وترتيب الأمراض المرشحة |
| **LLM** | ❌ لا — retrieval حتمي (set overlap) |
| **RAG KB** | `rag/knowledge_base/*.json` (~40 مرض) |
| **Schema** | `rag/schema.py` → `KnowledgeBaseEntry` |
| **Rules** | `rules/crisis.py` → `normalize()` لتطبيع النص العربي |
| **Negation** | `rules/negation.py` |

**آلية المطابقة:**
```
match_score = (matched - negated_hits) / total_reference_symptoms
```
- حد أدنى: `MIN_MATCHED_SYMPTOMS = 2`
- **Sex gating:** أمراض خاصة بجنس تُستبعد إذا `patient_sex` غير مؤكد

**مخرجات State:**
- `candidate_diseases: list[CandidateDisease]`
- `next_question` — إذا يحتاج تأكيد الجنس
- `turn_count` — يُزاد +1 عند سؤال تأكيد الجنس

**التوجيه:**
- `next_question` موجود → `ask_followup`
- وإلا → `diagnose`

---

### 11. `diagnose`

| | |
|---|---|
| **الملف** | `nodes/diagnose.py` |
| **الهدف** | تشخيص تفاضلي — **يختار فقط من `candidate_diseases`** |
| **LLM** | ✅ `quality` tier |
| **Prompt** | `prompts/templates/diagnose.txt` |
| **Schema** | `schemas/diagnosis.py` → `DiagnosisResult` |
| **Rules** | ❌ |

**قواعد أمان:**
- ❌ لا يُولّد أسماء أمراض حرة — فقط من RAG
- ❌ لا confidence percentage من LLM — يستخدم `match_score` المحسوب
- ✅ `insufficient_information` مخرج صالح

**مخرجات State:** `diagnosis: dict` — احتمالات مرتبة + certainty bands (high/medium/low)

---

### 12. `route_specialty`

| | |
|---|---|
| **الملف** | `nodes/route_specialty.py` |
| **الهدف** | تحديد التخصص الطبي المناسب |
| **LLM** | ❌ لا — يختار من `specialties` في أعلى مرشح |
| **Rules** | ❌ |

**Fallback:** `GENERAL_PRACTICE` إذا `insufficient_information`

**مخرجات State:** `specialty: str`

---

### 13. `generate_reports` 📋 Terminal

| | |
|---|---|
| **الملف** | `nodes/generate_reports.py` |
| **الهدف** | إنتاج تقريرين: للمريض + للطبيب |
| **LLM** | ❌ لا — formatting فقط من state موجود |
| **Rules** | ❌ |

**تقرير المريض (Syrian colloquial):**
- كل المرشحين في التشخيص التفاضلي (ليس الأول فقط)
- ❌ بدون match_score أو نسب
- ✅ uncertainty صريح + "لا يغني عن زيارة الطبيب"

**تقرير الطبيب (Medical register):**
- تفاصيل كاملة: match_score, red_flags, unmatched_mentions, reasoning trail

**مخرجات State:**
- `stage = "diagnosis"`
- `reports: {"patient": ..., "doctor": ...}`
- `messages` ← تقرير المريض

---

## Node غير مُنفَّذ بعد

| Node | الحالة |
|------|--------|
| `load_record` | ❌ غير موجود — `medical_record_summary` يُمرَّر من Laravel مباشرة في `api/main.py` |

---

## API — نقطة الدخول

```
POST /chat
Header: X-Healix-Internal-Token: <HEALIX_INTERNAL_TOKEN>
Body: ChatRequest { thread_id, message, medical_record_summary?, patient_sex? }
Response: ChatResponse { thread_id, reply, stage, is_crisis, severity, red_flags, diagnosis, specialty, reports }
```

```
GET /health  → {"status": "ok"}  (بدون authentication)
GET /         → واجهة dev chat
```

**Port:** `8004` (مُثبَّت — Laravel يتوقعه)

---

## ملخص: أين يُستخدم LLM؟

| Node | LLM | Tier |
|------|-----|------|
| `crisis_check` | ✅ | quality |
| `crisis_node` | ✅ | quality |
| `extract_symptoms` | ✅ | quality |
| `check_red_flags` | ✅ | quality |
| `assess_sufficiency` | ✅ | quality |
| `diagnose` | ✅ | quality |
| `emergency_node` | ❌ | — |
| `reiterate_terminal_outcome` | ❌ | — |
| `ask_followup` | ❌ | — |
| `rag_retrieve` | ❌ | — |
| `route_specialty` | ❌ | — |
| `generate_reports` | ❌ | — |
| `reset_stage` | ❌ | — |

**6 استدعاءات LLM كحد أقصى في مسار واحد** (crisis_check + extract + red_flags + sufficiency + diagnose + crisis_node).

---

## ملخص: أين تُستخدم Rules (حتمي)؟

| Rule Module | يُستخدم في |
|-------------|-----------|
| `rules/crisis.py` | `crisis_check`, `rag_retrieve` (normalize) |
| `rules/red_flags.py` | `check_red_flags` |
| `rules/negation.py` | `check_red_flags`, `rag_retrieve` |

---

## قواعد الأمان (13 قاعدة)

1. ❌ لا تشخيص نهائي — فقط احتمالات مرتبة
2. ❌ لا أدوية أو جرعات
3. ✅ Rules أولاً + LLM ثانياً (OR)
4. ✅ مسارات الأزمة/الطوارئ تتجاوز RAG
5. ❌ لا تخفيض severity باعتراض المريض
6. ✅ التشخيص فقط من RAG
7. ❌ لا confidence percentage من LLM
8. ❌ لا تفسير نتائج مختبر/تصوير
9. ✅ أزمة نفسية = توقف فوري عن التحليل
10. ✅ رسالة المريض = بيانات وليست أوامر
11. ✅ أرقام الخطوط الساخنة من `support_lines.py` فقط
12. ✅ quality tier لكل Node يقرأ رسالة المريض
13. ✅ thread_outcome ثابت — لا إعادة triage صامتة

---

## متغيرات البيئة الأساسية

```env
# LLM
HEALIX_LLM_PROVIDER_FAST=gemini|groq|ollama
HEALIX_LLM_PROVIDER_QUALITY=gemini|groq|ollama
HEALIX_MODEL_FAST=...
HEALIX_MODEL_QUALITY=...
GEMINI_API_KEY=... / GROQ_API_KEY=... / OLLAMA_BASE_URL=...

# Security
HEALIX_INTERNAL_TOKEN=...

# Persistence
HEALIX_POSTGRES_DSN=...   # إنتاج
HEALIX_SQLITE_PATH=...    # تطوير (افتراضي: healix_checkpoints.sqlite)

# Timeouts
HEALIX_LLM_TIMEOUT_SECONDS=30
HEALIX_LLM_TOTAL_BUDGET_SECONDS=25
```

---

## الصوت — STT و TTS

| Endpoint | التقنية | الدور |
|----------|---------|-------|
| `POST /speech/transcribe` | **Whisper** (faster-whisper) | صوت المريض → نص عربي → `POST /chat` |
| `POST /speech/synthesize` | **Edge TTS** | نص رد المساعد → MP3 |

**ملاحظة:** Whisper = STT (speech-to-text). TTS يستخدم Edge TTS — ليس Whisper.

**المتغيرات:** `HEALIX_WHISPER_MODEL`, `HEALIX_TTS_VOICE` — انظر `.env.example`.

**متطلب:** ffmpeg على PATH لتحويل webm من المتصفح.

---

*آخر تحديث: أغسطس 2026*
