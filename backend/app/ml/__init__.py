"""
Razorpay CloseLoop ML module.

Phase 4: Exception Intelligence — ML data contracts, feature extraction,
and training infrastructure.
"""

from app.ml.features import (
    ML_FEATURE_SCHEMA,
    extract_features,
    validate_features,
)
from app.ml.engineering import FeatureEngineer
from app.ml.classifier import (
    DatasetBuilder,
    ExceptionClassifier,
    ExceptionClassifierService,
    ModelArtifact,
    ModelEvaluator,
)
from app.ml.resolution import (
    ResolutionClassifier,
    ResolutionPredictor,
    ResolutionPrediction,
    ResolutionTrainingBuilder,
    EvidenceCompatibilityChecker,
    EXCEPTION_TO_RESOLUTION_MAP,
)

__all__ = [
    "ML_FEATURE_SCHEMA",
    "extract_features",
    "validate_features",
    "FeatureEngineer",
    "DatasetBuilder",
    "ExceptionClassifier",
    "ExceptionClassifierService",
    "ModelArtifact",
    "ModelEvaluator",
    "ResolutionClassifier",
    "ResolutionPredictor",
    "ResolutionPrediction",
    "ResolutionTrainingBuilder",
    "EvidenceCompatibilityChecker",
    "EXCEPTION_TO_RESOLUTION_MAP",
]
