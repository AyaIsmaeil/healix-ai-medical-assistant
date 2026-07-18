"""
Healix - Model Loader
تحميل وإدارة نموذج MARBERT المُدرّب على استخراج الأعراض (Token Classification / NER).

يُحمّل النموذج والـ Tokenizer مرّة واحدة فقط، ويختار الجهاز (GPU/CPU) تلقائياً.
التصميم قائم على الكائنات (Instance-based) لا على متغيّرات عامة، بحيث يُحقَن
الكائن الوحيد عبر ``app.state`` ويُمرّر بالاعتمادية (Dependency Injection).
"""

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.config import Config
from app.exceptions import ModelLoadError, ModelNotLoadedError

logger = logging.getLogger(__name__)
config = Config()

class ModelLoader:
    """
    مسؤول وحيد عن تحميل النموذج والـ Tokenizer وإتاحتهما.

    - يُحمّل مرّة واحدة (idempotent) بحماية Lock لسلامة الخيوط (thread-safe).
    - يختار GPU إن توفّر وسُمح به، وإلا CPU تلقائياً.
    - لا يحتفظ بأي حالة عامة على مستوى الوحدة (No global variables).
    """

    def __init__(
        self,
        model_source: Optional[str] = None,
        device: Optional[str] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        self._model_source = model_source or config.model_source()
        self._use_gpu = config.USE_GPU if use_gpu is None else use_gpu
        self._device = torch.device(device) if device else self._resolve_device()

        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._id2label: Optional[Dict[int, str]] = None
        self._checkpoint: Optional[Dict[str, Any]] = None

        self._loaded: bool = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # التحميل
    # ------------------------------------------------------------------
    def _resolve_device(self) -> torch.device:
        """اختيار الجهاز تلقائياً: GPU إن توفّر وسُمح به، وإلا CPU."""
        if self._use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load(self) -> "ModelLoader":
        """
        تحميل النموذج والـ Tokenizer مرّة واحدة.
        آمن للاستدعاء المتكرّر ومن خيوط متعدّدة.
        """
        if self._loaded:
            return self

        with self._lock:
            if self._loaded:  # فحص مزدوج بعد الحصول على القفل
                return self

            logger.info("🔄 تحميل نموذج استخراج الأعراض من: %s", self._model_source)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self._model_source, use_fast=True
                )
                if not getattr(tokenizer, "is_fast", False):
                    raise ModelLoadError(
                        "المطلوب Tokenizer سريع (Fast) لدعم محاذاة الكلمات (word_ids)."
                    )

                model = AutoModelForTokenClassification.from_pretrained(
                    self._model_source
                )
                model.to(self._device)
                model.eval()

                id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

                self._tokenizer = tokenizer
                self._model = model
                self._id2label = id2label
                self._checkpoint = self._fingerprint(id2label)
                self._loaded = True

                logger.info(
                    " تم تحميل النموذج | الجهاز: %s | عدد التصنيفات: %d | التصنيفات: %s | البصمة: %s",
                    self._device,
                    len(id2label),
                    list(id2label.values()),
                    self._checkpoint.get("weights_fingerprint"),
                )
                return self

            except ModelLoadError:
                raise
            except Exception as exc:  # noqa: BLE001 - نغلّفه في استثناء المجال
                logger.exception(" فشل تحميل النموذج")
                raise ModelLoadError(f"فشل تحميل النموذج: {exc}") from exc

    def _fingerprint(self, id2label: Dict[int, str]) -> Dict[str, Any]:
        """
        بصمة نقطة التحقّق المُحمَّلة لإثبات "أي نموذج مُحمَّل فعلاً" في نقطة الصحّة.
        تعتمد على حجم وتاريخ ملف الأوزان (فوري، يتغيّر عند استبدال النموذج).
        """
        info: Dict[str, Any] = {
            "source": self._model_source,
            "num_labels": len(id2label),
            "labels": list(id2label.values()),
            "weights_file": None,
            "weights_size_bytes": None,
            "weights_mtime": None,
            "weights_fingerprint": None,
        }
        weights = os.path.join(self._model_source, "model.safetensors")
        if os.path.isfile(weights):
            stat = os.stat(weights)
            info["weights_file"] = "model.safetensors"
            info["weights_size_bytes"] = stat.st_size
            info["weights_mtime"] = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            # بصمة مدمجة (حجم-تاريخ) تكفي لتمييز نقطة التحقّق دون قراءة 650MB.
            info["weights_fingerprint"] = f"{stat.st_size}-{int(stat.st_mtime)}"
        else:
            # مصدر بعيد (Hugging Face) — المُعرّف نفسه هو البصمة.
            info["weights_fingerprint"] = self._model_source
        return info

    # ------------------------------------------------------------------
    # الوصول للموارد
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise ModelNotLoadedError("النموذج غير مُحمّل. استدعِ load() أولاً.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> PreTrainedModel:
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._ensure_loaded()
        return self._tokenizer

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_loaded()
        return self._id2label

    @property
    def model_source(self) -> str:
        return self._model_source

    @property
    def checkpoint(self) -> Optional[Dict[str, Any]]:
        """بصمة نقطة التحقّق المُحمَّلة (لعرضها في نقطة الصحّة)."""
        return self._checkpoint
