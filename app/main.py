from fastapi import FastAPI
from sqlalchemy import text
from app.endpoints import extraction, chat, dashboard 
from app.database.database import engine, Base

app = FastAPI(title="RAG-Financial-Named-Entity-Recognition")

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

app.include_router(extraction.router, tags=["Ingestion"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(dashboard.router, tags=["Analytics"])
@app.get("/")
def read_root():
    return {"message": "System is running"}