from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.database import models
from app.schemas import ExtractionResponse
from app.services.service import GeminiService
from app.services.pdf_service import PDFService

router = APIRouter()
gemini_service = GeminiService()
pdf_service = PDFService()

@router.post("/upload_pdf", response_model=ExtractionResponse)
async def upload_financial_pdf(
    file: UploadFile = File(...), 
    db: AsyncSession = Depends(get_db)
):

    # Reading a PDF and extracting text
    content = await file.read()
    text = await pdf_service.extract_text_from_bytes(content)
    
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    new_doc = models.Document(
        filename=file.filename, 
        content=text
    )
    db.add(new_doc)
    
    await db.commit()
    await db.refresh(new_doc) 
    print(f"Document saved with ID: {new_doc.id}")

    # RAG Pipeline
    chunks = pdf_service.create_chunks(text)
    print(f"Created {len(chunks)} chunks.")
    
    for i, chunk_text in enumerate(chunks):
        vector = await gemini_service.get_embedding(chunk_text)
        new_chunk = models.DocumentChunk(
            document_id=new_doc.id, 
            content=chunk_text,
            embedding=vector
        )
        db.add(new_chunk)
        if i % 5 == 0: print(f"Chunk processed {i+1}/{len(chunks)}")
    
    # NER Pipeline
    extracted_data = await gemini_service.extract_entities(text[:30000])
    print(f"Found {len(extracted_data)} entities.")
    
    saved_entities = []
    for entity in extracted_data:
        new_entity = models.Entity(
            document_id=new_doc.id, 
            entity_type=entity.entity_type,
            value=entity.value
        )
        db.add(new_entity)
        saved_entities.append(new_entity)
        
    await db.commit()
    print("All data has been successfully saved to the database.")

    for entity in saved_entities:
        await db.refresh(entity)

    return ExtractionResponse(
        document_id=new_doc.id,
        extracted_at=new_doc.created_at,
        entities=saved_entities
    )