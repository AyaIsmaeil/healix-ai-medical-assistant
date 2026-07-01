"""
Unit tests for MARBERT symptom extraction service.

Tests:
- Model loading and initialization
- Symptom extraction
- Confidence threshold handling
- Error handling
- Edge cases
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.marbert_service import MARBERTService, SymptomExtractor
from app.utils.exceptions import ModelLoadingError, TextProcessingError, InferenceError


class TestMARBERTServiceInitialization:
    """Tests for MARBERT service initialization."""

    def test_service_singleton_pattern(self):
        """Test that MARBERT service follows singleton pattern."""
        service1 = MARBERTService()
        service2 = MARBERTService()
        
        assert service1 is service2

    def test_model_path_default(self):
        """Test that default model path is set correctly."""
        service = MARBERTService()
        expected_path = "ml_models/marbert_symptoms"
        assert str(service.model_path).endswith(expected_path) or service.model_path.name == "marbert_symptoms"

    def test_device_detection(self):
        """Test that device is correctly detected (CPU or CUDA)."""
        service = MARBERTService()
        
        # Should be either CPU or CUDA
        assert service.device.type in ("cpu", "cuda")
        
        # If CUDA available, should use it
        if torch.cuda.is_available():
            assert service.device.type == "cuda"
        else:
            assert service.device.type == "cpu"

    def test_model_loading_error_on_missing_path(self):
        """Test that ModelLoadingError is raised when model path doesn't exist."""
        with patch("app.services.marbert_service.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            
            with pytest.raises(ModelLoadingError):
                # Create new instance with invalid path
                service = object.__new__(MARBERTService)
                service.model_path = mock_path.return_value
                service.device = torch.device("cpu")
                service.model = None
                service.tokenizer = None
                service.symptom_labels = []
                service.metadata = {}
                service._initialized = False
                service._load_model()


class TestSymptomExtraction:
    """Tests for symptom extraction functionality."""

    @pytest.fixture
    def mock_marbert_service(self):
        """Create a mock MARBERT service for testing."""
        service = MARBERTService.__new__(MARBERTService)
        service.model_path = "ml_models/marbert_symptoms"
        service.device = torch.device("cpu")
        service.model = MagicMock()
        service.tokenizer = MagicMock()
        service.symptom_labels = [
            "Fever", "Cough", "Headache", "Sore Throat", "Fatigue",
            "Chills", "Muscle Aches", "Loss of Taste", "Loss of Smell", "Nausea"
        ]
        service.metadata = {
            "threshold": 0.3,
            "training_date": "2024-01-01",
        }
        service._initialized = True
        return service

    def test_symptom_extraction_success(self, mock_marbert_service):
        """Test successful symptom extraction."""
        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.8, -0.5, 0.6, -0.1, 0.9, -0.3, 0.4, -0.2, 0.1, 0.7]])
        mock_marbert_service.model.return_value = mock_output
        
        # Mock tokenizer
        mock_marbert_service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        result = mock_marbert_service.extract_symptoms("عندي حرارة وكحة")

        assert result["success"] is True
        assert "symptoms" in result
        assert len(result["symptoms"]) > 0
        assert "normalized_text" in result
        assert "processing_time_ms" in result
        assert "model_info" in result

    def test_symptom_confidence_threshold(self, mock_marbert_service):
        """Test that confidence threshold is applied correctly."""
        # Create mock output
        mock_output = Mock()
        # Probabilities: [0.9, 0.4, 0.2, 0.8, 0.1]
        mock_output.logits = torch.tensor([[2.0, 0.5, -1.0, 1.5, -2.0]])
        mock_marbert_service.model.return_value = mock_output
        
        mock_marbert_service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Extract with high threshold
        result_high = mock_marbert_service.extract_symptoms(
            "عندي حرارة",
            threshold=0.7
        )

        # Extract with low threshold
        result_low = mock_marbert_service.extract_symptoms(
            "عندي حرارة",
            threshold=0.1
        )

        # Higher threshold should result in fewer symptoms
        assert len(result_low["symptoms"]) >= len(result_high["symptoms"])

    def test_symptom_sorting_by_confidence(self, mock_marbert_service):
        """Test that symptoms are sorted by confidence (descending)."""
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.8, 0.5, 0.95, 0.3, 0.2]])
        mock_marbert_service.model.return_value = mock_output
        
        mock_marbert_service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        result = mock_marbert_service.extract_symptoms(
            "عندي حرارة",
            threshold=0.2
        )

        # Check that symptoms are sorted by confidence
        confidences = [s["confidence"] for s in result["symptoms"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidence_levels_assignment(self, mock_marbert_service):
        """Test that confidence levels are correctly assigned."""
        mock_output = Mock()
        # High: 0.9, Medium: 0.6, Low: 0.4
        mock_output.logits = torch.tensor([[2.0, 0.4, -0.4]])
        mock_marbert_service.model.return_value = mock_output
        
        mock_marbert_service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_marbert_service.symptom_labels = ["High", "Medium", "Low"]

        result = mock_marbert_service.extract_symptoms(
            "عندي حرارة",
            threshold=0.3
        )

        symptoms = result["symptoms"]
        
        # Find confidence levels
        for symptom in symptoms:
            if symptom["confidence"] >= 0.8:
                assert symptom["confidence_level"] == "High"
            elif symptom["confidence"] >= 0.5:
                assert symptom["confidence_level"] == "Medium"
            else:
                assert symptom["confidence_level"] == "Low"

    def test_empty_text_handling(self, mock_marbert_service):
        """Test that empty text after normalization raises error."""
        with pytest.raises(TextProcessingError):
            mock_marbert_service.extract_symptoms("   ")

    def test_text_normalization_applied(self, mock_marbert_service):
        """Test that Arabic text is normalized before inference."""
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.8, -0.5, 0.6]])
        mock_marbert_service.model.return_value = mock_output
        
        mock_marbert_service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        result = mock_marbert_service.extract_symptoms("عَنْدِي حَرَارَة")

        # Should contain normalized text
        assert "normalized_text" in result
        # Diacritics should be removed
        assert "َ" not in result["normalized_text"]

    def test_return_all_scores(self, mock_marbert_service):
        """Test that all symptom scores are returned when requested."""
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.8, -0.5, 0.6]])
        mock_marbert_service.model.return_value = mock_output
        
        mock_marbert_service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        result = mock_marbert_service.extract_symptoms(
            "عندي حرارة",
            return_probs=True
        )

        assert "all_symptoms_scores" in result
        assert len(result["all_symptoms_scores"]) == len(mock_marbert_service.symptom_labels)


class TestSymptomExtractorInterface:
    """Tests for high-level SymptomExtractor interface."""

    def test_extractor_initialization(self):
        """Test SymptomExtractor initialization."""
        # Should not raise error
        extractor = SymptomExtractor()
        assert extractor.service is not None

    def test_extractor_extract_method(self):
        """Test SymptomExtractor.extract() method."""
        extractor = SymptomExtractor()
        
        # Mock the service
        with patch.object(extractor.service, "extract_symptoms") as mock_extract:
            mock_extract.return_value = {
                "success": True,
                "symptoms": [{"name": "Fever", "confidence": 0.9}]
            }
            
            result = extractor.extract("عندي حرارة")
            
            assert result["success"] is True
            mock_extract.assert_called_once()

    def test_get_symptom_labels(self):
        """Test retrieving symptom labels."""
        extractor = SymptomExtractor()
        
        # Mock the service
        extractor.service.symptom_labels = ["Fever", "Cough", "Headache"]
        
        labels = extractor.get_symptom_labels()
        
        assert isinstance(labels, list)
        assert len(labels) == 3
        assert "Fever" in labels

    def test_get_model_info(self):
        """Test retrieving model information."""
        extractor = SymptomExtractor()
        
        info = extractor.get_model_info()
        
        assert "name" in info
        assert "initialized" in info
        assert "device" in info
        assert "num_symptoms" in info
        assert info["name"] == "MARBERT"


class TestErrorHandling:
    """Tests for error handling in MARBERT service."""

    def test_service_not_initialized_error(self):
        """Test that InferenceError is raised when service not initialized."""
        service = MARBERTService.__new__(MARBERTService)
        service._initialized = False
        service.model = None

        with pytest.raises(InferenceError):
            service.extract_symptoms("عندي حرارة")

    def test_inference_error_on_model_failure(self):
        """Test that InferenceError is raised on model inference failure."""
        service = MARBERTService.__new__(MARBERTService)
        service.model = Mock()
        service.model.side_effect = RuntimeError("CUDA out of memory")
        service.tokenizer = Mock()
        service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        service.symptom_labels = ["Fever"]
        service.metadata = {}
        service._initialized = True
        service.device = torch.device("cpu")

        with pytest.raises(InferenceError):
            service.extract_symptoms("عندي حرارة")


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_very_long_text(self):
        """Test processing of very long Arabic text."""
        service = MARBERTService.__new__(MARBERTService)
        service._initialized = True
        service.model = Mock()
        service.tokenizer = Mock()
        service.device = torch.device("cpu")
        
        # Generate long text
        long_text = "عندي حرارة وكحة " * 100  # Very long text
        
        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.8] * 10])
        service.model.return_value = mock_output
        
        service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        
        service.symptom_labels = [f"Symptom_{i}" for i in range(10)]
        service.metadata = {}

        # Should handle truncation gracefully
        result = service.extract_symptoms(long_text)
        assert result["success"] is True

    def test_mixed_language_text(self):
        """Test processing of mixed Arabic/English text."""
        service = MARBERTService.__new__(MARBERTService)
        service._initialized = True
        service.model = Mock()
        service.tokenizer = Mock()
        service.device = torch.device("cpu")
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.8] * 5])
        service.model.return_value = mock_output
        
        service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        
        service.symptom_labels = [f"Symptom_{i}" for i in range(5)]
        service.metadata = {}

        result = service.extract_symptoms("I have حرارة and كحة")
        assert result["success"] is True

    def test_multiple_consecutive_extractions(self):
        """Test that service can handle multiple consecutive extractions."""
        service = MARBERTService.__new__(MARBERTService)
        service._initialized = True
        service.model = Mock()
        service.tokenizer = Mock()
        service.device = torch.device("cpu")
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.8, 0.5, 0.3]])
        service.model.return_value = mock_output
        
        service.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        
        service.symptom_labels = ["Fever", "Cough", "Headache"]
        service.metadata = {}

        # Multiple extractions
        for _ in range(5):
            result = service.extract_symptoms("عندي حرارة")
            assert result["success"] is True
            assert len(result["symptoms"]) > 0