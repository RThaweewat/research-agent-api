"""
This module defines Pydantic models used for validating request payloads.
It includes the `QueryRequest` model for queries, which incorporates fields
for the question, user ID, and thread ID. Additionally, the `UploadRequest`
 model is used for specifying the file type during document uploads.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict

class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    config: Optional[Dict] = Field(None, description="Additional configuration for the query")

class UploadRequest(BaseModel):
    file_type: Optional[str] = Field("pdf", description="Type of file being uploaded")
