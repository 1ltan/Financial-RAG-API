from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.database import get_db
from app.database import models
from app.services.service import GeminiService
from pydantic import BaseModel

router = APIRouter()
gemini_service = GeminiService()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    # Transforming a question into a vector
    query_vector = await gemini_service.get_embedding(request.query)
    
    # Database search
    stmt = select(models.DocumentChunk).order_by(
        models.DocumentChunk.embedding.cosine_distance(query_vector)
    ).limit(5)
    
    result = await db.execute(stmt)
    relevant_chunks = result.scalars().all()
    
    if not relevant_chunks:
        return ChatResponse(answer="I couldn't find any relevant info in the database.", sources=[])

    # Context construction for an LLM
    context_texts = [chunk.content for chunk in relevant_chunks]
    
    # Answer generation
    answer = await gemini_service.generate_rag_answer(request.query, context_texts)
    
    return ChatResponse(
        answer=answer,
        sources=context_texts
    )