from typing import Optional
from pydantic import BaseModel

class VideoIngestRequest(BaseModel):
    url: str
    source_id: Optional[str] = None

class PDFIngestRequest(BaseModel):
    text: str
    filename: str
    source_id: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    source_id: Optional[str] = None

class VectorRequest(BaseModel):
    text: str

class GenerateRequest(BaseModel):
    question: str
    context: str