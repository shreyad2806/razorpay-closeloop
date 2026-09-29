"""
Ingestion — the normalization boundary between provider and database.

Flow (architecture section 14):

    Provider DTOs -> validate -> normalize -> ordered idempotent upsert -> DB

Ingestion answers only "what financial records did the provider report?".
Reconciliation, exception detection and everything downstream belong to later
phases and are deliberately absent here.
"""

from app.ingestion.result import IngestionError, IngestionResult
from app.ingestion.service import IngestionService

__all__ = ["IngestionService", "IngestionResult", "IngestionError"]
