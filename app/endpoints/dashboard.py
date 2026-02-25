from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database.database import get_db
from app.database import models
from pydantic import BaseModel

router = APIRouter()

class KPIData(BaseModel):
    metric: str
    value: str

class CompanyReport(BaseModel):
    document_name: str
    upload_date: str
    financials: list[KPIData]

@router.get("/dashboard", response_model=list[CompanyReport])
async def get_financial_dashboard(db: AsyncSession = Depends(get_db)):
    stmt = select(models.Document).options(selectinload(models.Document.entities)).order_by(models.Document.created_at.desc())
    result = await db.execute(stmt)
    documents = result.scalars().all()
    
    dashboard_data = []
    
    for doc in documents:
        kpis = []
        for entity in doc.entities:
            kpis.append(KPIData(metric=entity.entity_type, value=entity.value))
            
        dashboard_data.append(CompanyReport(
            document_name=doc.filename or "Unknown Report",
            upload_date=doc.created_at.strftime("%Y-%m-%d %H:%M"),
            financials=kpis
        ))
        
    return dashboard_data