from pydantic import BaseModel
from typing import List
from datetime import datetime

class EntityBase(BaseModel):
    entity_type: str
    value: str

class EntityCreate(EntityBase):
    pass

class EntityResponse(EntityBase):
    id: int
    class Config:
        from_attributes = True

class ExtractionResponse(BaseModel):
    document_id: int
    extracted_at: datetime
    entities: List[EntityResponse]
    class Config:
        from_attributes = True