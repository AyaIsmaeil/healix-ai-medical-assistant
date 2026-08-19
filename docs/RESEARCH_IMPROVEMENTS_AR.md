# تحسينات Healix المبنية على أدلة علمية (Evidence-Based Improvement Plan)

> منهجية هالملف: كل نقطة هون مبنية على بحث علمي حقيقي (بحثت عنه فعليًا، مو من الذاكرة)، مربوطة بمكان محدد بكود Healix الحالي. بفصل دايمًا بين **"البحث بيقول"** (اقتباس/تلخيص من مصدر حقيقي) و**"توصيتي لـHealix"** (رأيي أنا كتطبيق عملي) — منعًا لخلط الاثنين متل ما هو معتمد بمعيار المشروع.
>
> 🔴 أولوية عالية | 🟡 أولوية متوسطة | 🟢 أولوية منخفضة (تحسين إضافي)

---

## 1. الفرز (Triage) أهم من دقة التشخيص — وعندك فجوة حقيقية هون

**البحث بيقول**: مراجعتان منهجيتان حديثتان (npj Digital Medicine 2022، JMIR 2023) فحصتا أدوات فحص الأعراض الرقمية، ولقيتا إن **دقة التشخيص الأولي كانت منخفضة جدًا (19-37.9%)**، بينما **دقة الفرز (triage) كانت أعلى بكثير (48.8-90.1%)**. يعني الأنظمة المشابهة لـHealix بتنجح أكتر بـ"قديش الحالة مستعجلة" أكتر ما بتنجح بـ"شو المرض بالضبط".

**كمان**: ورقة بحثية حديثة (arXiv 2607.28677، "Reasoning in Real World Clinical Care") بتحدد بالضبط شو الفرز الآمن يحتاج: **توسيع** قائمة الاحتمالات (مو تضييقها) لما المعلومات ناقصة، **البحث الفعلي** عن red flags مفقودة بدل انتظارها، **خفض** عتبة التصعيد عند الشك، وتأجيل أي حكم لحد ما توصل معلومات كافية.

**طلوعه بحالة Healix**: 🔴 هاد بالضبط الفجوة يلي حددناها سابقًا — **Urgency/Triage غير موجود إطلاقًا** (لا حتى كـplaceholder). عندك حاليًا فقط binary: طوارئ حقيقية (`emergency_node`) مقابل كل شي تاني. لا يوجد تدرّج (عاجل/شبه عاجل/غير عاجل).

**توصيتي**: 🔴 **أولوية عالية**. هاد أعلى قيمة ممكن تضيفيها من ناحية "شو بيخلي الشات بوت عنجد أذكى وأأمن" — لأنه بالضبط النقطة يلي البحث بيقول إنها أكتر شي بينجح فيه هيك أنظمة. اقتراح عملي بسيط (مو معقّد): إضافة حقل `urgency` بـ3-4 مستويات (فوري/عاجل خلال ساعات/عادي) يُحسَب من نفس مدخلات `check_red_flags` الموجودة أصلًا (مو نظام جديد من الصفر) — يعني rule-based بسيط زي `red_flags.py` بالضبط، بس بمخرج متدرّج بدل binary.

---

## 2. تحديد "الشكوى الرئيسية" عند تعدد الأعراض — الحل الأبسط مثبت عمليًا

**البحث بيقول**: بحث حديث (npj Digital Medicine 2025، تصنيف شكاوى مرضى بـLLM على مستوى أنظمة صحية أمريكية) واجه نفس المشكلة يلي وصفتيها (Clinical Priority Engine): مريض بيذكر أكتر من عرض بنفس الوقت. **الحل يلي اعتمدوه عمليًا مو خوارزمية ترتيب معقّدة (acuity/severity/recency)** — لما النموذج يرجّع أكتر من عرض محتمل، بيعرضوهم كلهم عالمريض **ويخلوه هو يختار شو "أكتر شي مزعجه"** مباشرة.

**طلوعه بحالة Healix**: 🔴 عندك فعلاً **صفر** آلية لهاد — النظام بيعامل كل الأعراض المستخرجة كمجموعة واحدة بلا أي ترتيب أولوية.

**توصيتي**: 🟡 **أولوية متوسطة، لكن حل بسيط جدًا مقارنة بالوصف الأصلي**. بدل بناء "Clinical Priority Engine" بمعايير acuity/severity/emphasis/recency (تعقيد كبير وبيانات مو متوفرة لمعايرته)، الحل المثبت بحثيًا أبسط بكتير: لو المريض ذكر أكتر من عرض بنفس الرسالة الأولى، `assess_sufficiency` (أو سؤال متابعة مخصص) يسأله مباشرة "شو أكتر عرض مزعجك هلق؟" — سؤال واحد بسيط، مو محرك حساب. هاد بينسجم تمامًا مع فلسفة Healix الحالية (قرارات بسيطة وقابلة للتتبع، مو ML معقّد بلا داعي).

---

## 3. كشف الأزمة النفسية: مشكلة تصميمية موجودة عندك، مو بس مشكلة "الأنماط placeholder"

**البحث بيقول**: بحث حديث جدًا (medRxiv 2026، تصميم أنظمة أمان للشات بوتات النفسية) بيحدد مبدأ مهم: **"overly protective guardrails cause many systems to terminate conversations abruptly after issuing generic hotline instructions"** — وهاد **يتعارض** مع مبادئ التدخل بالأزمة النفسية المُثبتة، لأن **الاستمرار بالتفاعل (sustained engagement) غالبًا ضروري للتهدئة (de-escalation)**، مو القطع الفوري.

**طلوعه بحالة Healix**: 🟡 هاي فجوة تصميمية **إضافية** لقيتها هلق، مو بس موضوع `rules/crisis.py` (يلي already موثّق كـplaceholder). حتى لو ملّينا الأنماط الحقيقية وصلحنا `support_lines.py` بأرقام حقيقية، **`crisis_node.py` بالتصميم الحالي بيرسل رسالة وحدة وبينتهي فورًا** (`stage="crisis"`, `messages` تُضاف، الـturn بينتهي بـEND) — بالضبط النمط يلي البحث بيحذّر منه.

**توصيتي**: 🔴 **أولوية عالية جدًا، وأهم من ملء أنماط crisis.py لحالها**. حتى مع طبقة كشف مثالية، تصميم "رسالة وحدة وخلص" مو الأنسب سريريًا. اقتراح: تعديل `crisis_node` (أو إضافة node بعده) يخلي المحادثة مستمرة — المريض يقدر يرد، والنظام يفضل بوضع "أزمة" (`thread_outcome="crisis"` already بيضمن هاد عبر `reiterate_terminal_outcome`) بدل ما ينتهي كليًا. هاد تغيير معماري متوسط الحجم، مو بس تعديل نص.

---

## 4. RAG بتقلل الهلوسة، بس جودة الاسترجاع هي الاختناق — مو الـprompt

**البحث بيقول**: مسح شامل حديث (arXiv 2505.01146، RAG بالطب الحيوي) بيأكد إن RAG فعليًا بتقلل هلوسة النماذج اللغوية بربط الإجابة بمصدر موثوق. **لكن**: "retrieval relevance on medical benchmarks can be as low as 22%... **the primary bottleneck lies not in generation but in retrieval quality**" — يعني المشكلة الأكبر بأنظمة RAG الطبية مو بصياغة الـprompt، هي بجودة/حجم قاعدة المعرفة نفسها.

**طلوعه بحالة Healix**: 🟢 هاد **يبرر بالضبط** قرارك بالجلسة الماضية إنك تركزي على البرومبتات أول، بس بيقول كمان إن أكبر عائد مستقبلي رح يكون بتوسيع الـ49 مرض، مو بإعادة صياغة `diagnose.txt` أكتر.

**توصيتي**: 🟢 **أولوية منخفضة هلق** (already عندك خطة توسيع بيانات بمكان تاني)، بس سجليها كتأكيد بحثي: أي وقت مستقبلي بدك تستثمريه بتحسين دقة التشخيص، حطيه بتوسيع `rag/knowledge_base/` مو بتعقيد الـprompt.

---

## 5. معايرة الثقة (Confidence Calibration) — فجوة معروفة بكل المجال، مو بس عندك

**البحث بيقول**: مسح حديث (survey 2025 عن Uncertainty Quantification بالـLLMs) بيأكد إن **"LLMs tend to generate plausible but false answers when uncertain rather than admitting inability"** — وإن معايرة ثقة حقيقية بتحتاج إما تدريب إضافي (fine-tuning/RL) أو طرق white-box معقّدة (احتمالات التوكن، entropy) — مو حل بسيط بمستوى الـprompt.

**طلوعه بحالة Healix**: 🟡 الملاحظة يلي كانت عندك (`_certainty_band` بـ`diagnose.py` مبني على `match_score` بسيط، مو معايَر بـGround Truth) **صحيحة ومطابقة تمامًا للفجوة العامة بالمجال كله** — مو تقصير خاص بـHealix. هاد بالحقيقة نقطة قوة تقدري تقوليها للجنة: "إحنا واعيين إن الثقة الحالية heuristic بسيط، وهاد نفس التحدي المفتوح بكل أبحاث الـLLM الطبي حاليًا، مو تقصير تصميمي."

**توصيتي**: 🟢 **أولوية منخفضة للتنفيذ الفعلي** (يحتاج بيانات ground-truth حقيقية غير متوفرة أصلًا)، لكن 🟡 **أولوية متوسطة للتوثيق**: أضيفي جملة صريحة بتقرير الطبيب أو بالوثائق توضح إن `match_score`/الدرجات النوعية heuristic غير معايَر، مو نتيجة نموذج مُدرَّب على دقة حقيقية — شفافية أكتر مع الطبيب المستخدم.

---

## 6. Safety Netting — تقنية سريرية موجودة بالطب التقليدي، وناقصة عند Healix

**البحث بيقول**: "Safety netting" مفهوم موثّق بمراجعات (BJGP) كتقنية استشارة أساسية: إعطاء المريض **معلومات صريحة ومحددة** عن "شو المتوقع/الجدول الزمني للتعافي، شو العلامات المقلقة يلي لازم يرجع فيها، وكيف/متى يطلب مساعدة" — مهمة بشكل خاص للأطفال، والحالات النفسية، وتعدد الأمراض.

**طلوعه بحالة Healix**: 🟡 تقرير المريض الحالي (`generate_reports.py`) فيه "تنويه عدم يقين" عام + "راجع طبيب"، لكن **مافيه safety netting منظّم ومحدد** (زي: "لو زادت الحرارة فوق كذا، أو ظهر كذا، روح فورًا للطوارئ" — مبني على المرض المرشّح تحديدًا).

**توصيتي**: 🟡 **أولوية متوسطة، وتكلفتها منخفضة نسبيًا**. ممكن تُضاف كحقل اختياري بملفات `rag/knowledge_base/*.json` (مثلًا `safety_netting_ar`: نص قصير مخصص لكل مرض)، ويُعرض بتقرير المريض تلقائيًا عبر `generate_reports.py` — تعديل بسيط نسبيًا، وقيمة سريرية حقيقية وموثّقة بحثيًا.

---

## 7. قابلية القراءة والثقة بالمحتوى الموجّه للمريض

**البحث بيقول**: دراسات حديثة (2025) بتأكد إن النصوص المبسّطة **لازم توصل تعاطف ومسؤولية**، مو بس تبسيط لغوي — "texts that are overly generic, impersonal, or dismissive of complexity may erode trust". وبحث منفصل عن ثقة المرضى بالـAI الطبي بيأكد إن **الشفافية حول الإشراف البشري ("يحتاج طبيب حقيقي")** بترفع الثقة بنسبة 14-19%.

**طلوعه بحالة Healix**: 🟢 تقرير المريض الحالي أصلًا بالعامية السورية، وفيه تذكير دائم بزيارة الطبيب — منسجم غالبًا مع هاد. نقطة تحسين محتملة: التأكد إن الصياغة مو "باردة/آلية" بس هاد يحتاج مراجعة بشرية فعلية للنص، مو تحليل كود.

**توصيتي**: 🟢 **أولوية منخفضة**، مراجعة نوعية لصياغة `_build_patient_report` بـ`generate_reports.py` بعين "هل هاد بيحس المريض إنه مسموع" — مو تغيير بنيوي.

---

## الترتيب النهائي حسب الأولوية والقيمة مقابل الجهد

| # | التحسين | الأولوية | الجهد التقريبي | الملفات المتأثرة |
|---|---|---|---|---|
| 1 | طبقة Urgency/Triage متدرّجة (rule-based بسيطة) | 🔴 عالية | متوسط | `state.py`, node جديد أو تعديل `check_red_flags.py` |
| 2 | إعادة تصميم `crisis_node` لاستمرارية التفاعل بدل رسالة وحدة | 🔴 عالية | متوسط-كبير | `nodes/crisis_node.py`, `graph.py` |
| 3 | سؤال "شو أكتر شي مزعجك" عند تعدد الأعراض | 🟡 متوسطة | صغير | `nodes/assess_sufficiency.py` أو `prompts/templates/assess_sufficiency.txt` |
| 4 | Safety netting مخصص لكل مرض بتقرير المريض | 🟡 متوسطة | صغير-متوسط | `rag/schema.py`, `rag/knowledge_base/*.json`, `generate_reports.py` |
| 5 | توثيق صريح إن الثقة heuristic غير معايَر | 🟡 متوسطة (توثيق فقط) | صغير جدًا | `docs/`, تعليق بالتقرير |
| 6 | توسيع قاعدة معرفة RAG (أكثر من تحسين prompt) | 🟢 منخفضة (مؤجّلة أصلًا) | كبير | `rag/knowledge_base/` |
| 7 | مراجعة نوعية لتعاطف صياغة تقرير المريض | 🟢 منخفضة | صغير | `generate_reports.py` |

---

## المصادر الكاملة

1. [The diagnostic and triage accuracy of digital and online symptom checker tools: a systematic review — npj Digital Medicine](https://www.nature.com/articles/s41746-022-00667-w)
2. [Triage and Diagnostic Accuracy of Online Symptom Checkers: Systematic Review — JMIR 2023](https://www.jmir.org/2023/1/e43803)
3. [Reasoning in Real World Clinical Care: Why Large Language Models Are Not Yet Safe for Autonomous Clinical Decision Support — arXiv](https://arxiv.org/abs/2607.28677)
4. [Transforming Health Care Through Chatbots for Medical History-Taking and Future Directions: Comprehensive Systematic Review — JMIR Medical Informatics 2024](https://medinform.jmir.org/2024/1/e56628)
5. [Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey](https://www.researchgate.net/publication/394261390_Uncertainty_Quantification_and_Confidence_Calibration_in_Large_Language_Models_A_Survey)
6. [Suicide- and crisis-risk detection using large language models in mental-health chatbots — medRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.01.12.26343914v1.full)
7. [Safety netting for primary care: evidence from a literature review — British Journal of General Practice](https://bjgp.org/content/69/678/e70)
8. [LLM enabled classification of patient self-reported symptoms and needs in health systems across the USA — npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-01779-9)
9. [Retrieval-Augmented Generation in Biomedicine: A Survey of Technologies, Datasets, and Clinical Applications — arXiv](https://arxiv.org/html/2505.01146v1)
10. [Key Information Influencing Patient Decision-Making About AI in Health Care: Survey Experiment Study](https://pubmed.ncbi.nlm.nih.gov/41525463/)

---

*ملاحظة منهجية: هاي قائمة أولى مبنية على بحث ويب مباشر بجلسة واحدة، مو مراجعة أدبيات شاملة (systematic literature review) بمعايير أكاديمية صارمة. كافية لتوجيه قرارات تطوير عملية، لكن لو ناوية تستشهدي فيها بمرجع أكاديمي رسمي (بحث تخرج)، لازم توثيق APA/IEEE كامل لكل مصدر، مو رابط فقط.*
