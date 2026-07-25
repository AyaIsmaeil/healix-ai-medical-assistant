"""
Healix - LLM Feature Extractor
يملأ الحقول الستة التي عجز عنها ``RuleBasedFeatureExtractor`` عبر الـLLM.

يعيد استخدام منفذ ``LLMProvider`` الحالي بلا أي تعديل عليه — نفس نمط
``ConversationService``. غرضه محدود بمرحلة ٣.١: ملء الحقول الناقصة فقط،
بلا تحسين تعليمات متقدّم.

تحسين أداء/تكلفة متعمّد: لو ``unresolved_fields`` فاضية (الاستخراج القاعدي
حلّ كل شيء)، لا يُستدعى الـLLM إطلاقاً — يعيد نتيجة فارغة مباشرة.
"""

from __future__ import annotations

import logging
from typing import List

from app.domain.feature_extraction_rules import RuleExtractionResult
from app.domain.ports import LLMProvider
from app.exceptions import FeatureExtractionError
from app.parsing.assessment_extraction_parser import parse_extraction_result
from app.prompts.assessment_extraction_builder import AssessmentExtractionPromptBuilder

logger = logging.getLogger(__name__)


class LLMFeatureExtractor:
    """يستدعي الـLLM لملء حقول التقييم الناقصة فقط، مع تخطٍّ عند عدم الحاجة."""

    def __init__(self, provider: LLMProvider, prompt_builder: AssessmentExtractionPromptBuilder):
        self._provider = provider
        self._prompts = prompt_builder

    def extract(
        self, raw_messages: List[str], unresolved_fields: List[str]
    ) -> RuleExtractionResult:
        if not unresolved_fields:
            logger.debug("لا حقول ناقصة — تخطّي استدعاء LLM الاستخراج.")
            return RuleExtractionResult()

        system_prompt = self._prompts.system_prompt()
        user_prompt = self._prompts.extraction_prompt(raw_messages, unresolved_fields)

        try:
            completion = self._provider.generate(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - فشل ناعم، لا ينهار التقييم كاملاً
            logger.warning("فشل استدعاء LLM استخراج الميزات: %s", exc)
            return RuleExtractionResult()

        try:
            return parse_extraction_result(completion.text)
        except FeatureExtractionError as exc:
            logger.warning("فشل تحليل مخرجات استخراج الميزات: %s", exc)
            return RuleExtractionResult()
