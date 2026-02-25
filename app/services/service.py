import google.generativeai as genai
import json
from app.config import settings
from app.schemas import EntityCreate

genai.configure(api_key=settings.GEMINI_API_KEY)

class GeminiService:
    def __init__(self):
        self.ner_model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        self.chat_model = genai.GenerativeModel("gemini-2.5-flash")
        self.embedding_model = "models/gemini-embedding-001"

    async def get_embedding(self, text: str) -> list[float]:
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']
            if len(embedding) > 768:
                embedding = embedding[:768]
            return embedding
        except Exception as e:
            print(f"Vector Error: {e}")
            return [0.0] * 768

    async def extract_entities(self, text: str) -> list[EntityCreate]:
        prompt = f"""
        You are a professional financial data extractor. Analyze the text and extract ONLY the following Key Performance Indicators (KPIs) for the current quarter/year mentioned:
        
        1. REVENUE (Total Revenue, Net Sales)
        2. NET_INCOME (Net Profit, Net Earnings)
        3. EPS (Earnings Per Share - Diluted)
        4. OPERATING_MARGIN (Operating Income Margin)
        5. CASH (Cash and Cash Equivalents)

        Return a JSON object with a key "entities". Each entity must have:
        - "entity_type": One of [REVENUE, NET_INCOME, EPS, OPERATING_MARGIN, CASH]
        - "value": The value found (e.g., "$18.1 billion", "4.55", "35%")
        
        If a metric is not found, do not invent it.
        
        Text excerpt: 
        "{text[:25000]}" 
        """
        
        try:
            response = await self.ner_model.generate_content_async(prompt)
            result = json.loads(response.text)
            entities = []
            
            if "entities" in result:
                for item in result["entities"]:
                    e_type = item.get("entity_type")
                    e_value = item.get("value")
                    
                    if e_type and e_value:
                        entities.append(EntityCreate(
                            entity_type=str(e_type).upper(),
                            value=str(e_value)
                        ))
            return entities
        except Exception as e:
            print(f"NER Error: {e}")
            return []

    async def generate_rag_answer(self, query: str, context_chunks: list[str]) -> str:
        context_str = "\n\n".join(context_chunks)
        prompt = f"""
        Answer based on context. If unknown, say so.
        Context: {context_str}
        Question: {query}
        """
        response = await self.chat_model.generate_content_async(prompt)
        return response.text