"""
MARBERT Service for Arabic Medical Text Processing.

Handles:
- Model loading from trained checkpoints
- Text tokenization and encoding
- Symptom extraction inference
- Confidence score calculation
- Structured output generation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.app_config import config
from app.utils.exceptions import (
    ModelLoadingError,
    TextProcessingError,
    InferenceError,
)
from app.utils.arabic_normalizer import normalize_arabic_text

logger = logging.getLogger(__name__)


class MARBERTService:
    """
    Service for loading and managing MARBERT model.
    
    Handles model initialization, caching, and resource management.
    """

    _instance: Optional["MARBERTService"] = None
    _lock = False

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for model service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize MARBERT service.
        
        Args:
            model_path: Path to trained model directory.
                       Defaults to config.settings.marbert.path
        """
        # Skip re-initialization if already initialized
        if hasattr(self, "_initialized"):
            return

        self.model_path = Path(model_path or config.get("models.marbert.path", "ml_models/marbert_symptoms"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.symptom_labels = []
        self.metadata = {}
        
        self._initialized = False
        self._load_model()

    def _load_model(self) -> None:
        """
        Load trained MARBERT model and tokenizer.
        
        Raises:
            ModelLoadingError: If model loading fails.
        """
        try:
            if not self.model_path.exists():
                raise ModelLoadingError(
                    model_name="MARBERT",
                    details={"path": str(self.model_path), "reason": "Path does not exist"}
                )

            logger.info(f"Loading MARBERT model from {self.model_path}")

            # Load model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model moved to device: {self.device}")

            # Load metadata
            self._load_metadata()

            self._initialized = True
            logger.info("✅ MARBERT service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to load MARBERT model: {str(e)}")
            raise ModelLoadingError(
                model_name="MARBERT",
                details={"error": str(e), "path": str(self.model_path)}
            )

    def _load_metadata(self) -> None:
        """
        Load training metadata (symptom labels, thresholds, etc.).
        
        Looks for metadata.json or training_meta.json in model directory.
        """
        metadata_paths = [
            self.model_path / "metadata.json",
            self.model_path / "training_meta.json",
            self.model_path / "labels.json",
        ]

        for metadata_path in metadata_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    logger.info(f"Loaded metadata from {metadata_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

        # Extract symptom columns if available
        if "symptom_columns" in self.metadata:
            self.symptom_labels = self.metadata["symptom_columns"]
        elif "labels" in self.metadata:
            self.symptom_labels = self.metadata["labels"]
        else:
            # Fallback: use model's number of labels
            num_labels = self.model.config.num_labels
            self.symptom_labels = [f"Symptom_{i}" for i in range(num_labels)]
            logger.warning(f"No symptom labels found in metadata. Using default labels: {num_labels}")

    def is_ready(self) -> bool:
        """Check if service is ready for inference."""
        return self._initialized and self.model is not None

    def extract_symptoms(
        self,
        text: str,
        threshold: Optional[float] = None,
        return_probs: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract symptoms from Arabic medical text using MARBERT.
        
        Args:
            text: Arabic medical text to process.
            threshold: Confidence threshold (0-1). Defaults to config value.
            return_probs: If True, return probability scores for all symptoms.
        
        Returns:
            Dictionary containing:
            - symptoms: List of detected symptoms with confidence
            - raw_scores: Raw model outputs (if return_probs=True)
            - processing_time_ms: Inference time
            - model_info: Model version and metadata
        
        Raises:
            TextProcessingError: If text processing fails.
            InferenceError: If inference fails.
        """
        if not self.is_ready():
            raise InferenceError(
                message="MARBERT service not initialized",
                model_type="MARBERT"
            )

        try:
            # Normalize Arabic text
            normalized_text = normalize_arabic_text(text)
            
            if not normalized_text:
                raise TextProcessingError(
                    message="Text is empty after normalization",
                    details={"original_length": len(text)}
                )

            logger.debug(f"Processing text: {normalized_text[:100]}...")

            # Get threshold
            if threshold is None:
                threshold = float(self.metadata.get("threshold", config.get("inference.symptom_confidence_threshold", 0.3)))

            # Tokenize
            import time
            start_time = time.time()

            inputs = self.tokenizer(
                normalized_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            processing_time = (time.time() - start_time) * 1000

            # Extract symptoms above threshold
            symptoms = self._format_symptoms(probs, threshold)

            result = {
                "success": True,
                "text": text,
                "normalized_text": normalized_text,
                "symptoms": symptoms,
                "confidence_threshold": threshold,
                "processing_time_ms": round(processing_time, 2),
                "model_info": {
                    "name": "MARBERT",
                    "num_labels": len(self.symptom_labels),
                    "training_date": self.metadata.get("training_date", "unknown"),
                },
            }

            # Add raw scores if requested
            if return_probs:
                result["all_symptoms_scores"] = self._format_all_symptoms(probs)

            logger.info(f"Extracted {len(symptoms)} symptoms from text")
            return result

        except TextProcessingError:
            raise
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise InferenceError(
                message=f"Symptom extraction failed: {str(e)}",
                model_type="MARBERT",
                details={"error": str(e)}
            )

    def _format_symptoms(
        self,
        probs: np.ndarray,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Format detected symptoms into structured output.
        
        Args:
            probs: Probability scores from model.
            threshold: Confidence threshold.
        
        Returns:
            List of symptom dictionaries with confidence scores.
        """
        symptoms = []

        for idx, prob in enumerate(probs):
            if prob >= threshold:
                symptom_name = self.symptom_labels[idx] if idx < len(self.symptom_labels) else f"Symptom_{idx}"
                
                # Determine confidence level
                if prob >= 0.8:
                    confidence_level = "High"
                elif prob >= 0.5:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"

                symptoms.append({
                    "canonical": symptom_name,
                    "english": symptom_name,  # Can be mapped to English later
                    "confidence": round(float(prob), 4),
                    "confidence_level": confidence_level,
                })

        # Sort by confidence descending
        symptoms.sort(key=lambda x: x["confidence"], reverse=True)
        return symptoms

    def _format_all_symptoms(self, probs: np.ndarray) -> List[Dict[str, Any]]:
        """
        Format all symptoms with their confidence scores.
        
        Args:
            probs: Probability scores from model.
        
        Returns:
            List of all symptoms with scores.
        """
        all_symptoms = []

        for idx, prob in enumerate(probs):
            symptom_name = self.symptom_labels[idx] if idx < len(self.symptom_labels) else f"Symptom_{idx}"
            all_symptoms.append({
                "name": symptom_name,
                "score": round(float(prob), 4),
            })

        return sorted(all_symptoms, key=lambda x: x["score"], reverse=True)

    def __del__(self):
        """Cleanup on deletion."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer


class SymptomExtractor:
    """
    High-level interface for symptom extraction.
    
    Wraps MARBERTService for cleaner API.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize symptom extractor.
        
        Args:
            model_path: Path to trained model.
        """
        self.service = MARBERTService(model_path)

    def extract(
        self,
        text: str,
        confidence_threshold: Optional[float] = None,
        include_all_scores: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract symptoms from text.
        
        Args:
            text: Arabic medical text.
            confidence_threshold: Threshold for symptom detection.
            include_all_scores: Include scores for all symptoms.
        
        Returns:
            Extraction results.
        """
        return self.service.extract_symptoms(
            text=text,
            threshold=confidence_threshold,
            return_probs=include_all_scores,
        )

    def get_symptom_labels(self) -> List[str]:
        """Get list of all symptom labels."""
        return self.service.symptom_labels

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "MARBERT",
            "initialized": self.service.is_ready(),
            "device": str(self.service.device),
            "num_symptoms": len(self.service.symptom_labels),
            "metadata": self.service.metadata,
        }