"""
ModelPrediction - lineage row for every ML output.

NEW in CloseLoop 2.0 (architecture section 5). Today the classifier's output
exists only inside a workflow payload, so a prediction cannot be traced back to
the model version, feature schema or input hash that produced it. This table
gives every prediction an addressable row.

No training, feature engineering or inference is implemented in Phase 1 (task
section 15); this is the persistence model only.
"""

from sqlalchemy import Column, ForeignKey, Index, Numeric, String

from app.database.database import Base
from app.models.types import JSONType, TimestampType, utcnow


class ModelPrediction(Base):
    """One persisted model prediction for an exception."""

    __tablename__ = "model_predictions"

    id = Column(String, primary_key=True)

    exception_id = Column(
        String,
        ForeignKey("exceptions.id", ondelete="RESTRICT"),
        nullable=False,
    )

    model_name = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=False)
    feature_schema_version = Column(String(32), nullable=False)

    predicted_type = Column(String(48), nullable=False)

    # class label -> probability
    probabilities = Column(JSONType, nullable=True)

    confidence = Column(Numeric(5, 4), nullable=True)

    # Hash of the feature vector, so identical inputs are detectable.
    features_hash = Column(String(128), nullable=True)

    created_at = Column(TimestampType, nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_model_predictions_exception", "exception_id"),
        Index("ix_model_predictions_model", "model_name", "model_version"),
        Index("ix_model_predictions_features_hash", "features_hash"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<ModelPrediction id={self.id!r} exception_id={self.exception_id!r} "
            f"{self.model_name}@{self.model_version} -> {self.predicted_type!r}>"
        )
