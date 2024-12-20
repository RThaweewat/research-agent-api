"""
Response models for API interactions are defined in this module.
It includes the `QueryResponse` model for returning answers to queries,
as well as `UploadResponse` and `DocumentReference` models for
document-related operations. An `ErrorResponse` model is also included
to standardize error handling.
"""

from pydantic import BaseModel
from typing import List, Optional

class DocumentReference(BaseModel):
    source: str
    relevance_score: float
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    references: List[DocumentReference]
    thread_id: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    file_count: int

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
