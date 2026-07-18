import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.database import LocalVectorDB
from src.config import config

app = FastAPI(title="Qdrant MCP Tool Gateway")
db = LocalVectorDB()

class SearchRequest(BaseModel):
    query_vector: List[float]
    limit: int = 3

@app.post("/tools/search_icd10_vectors")
def search_icd10_vectors(payload: SearchRequest):
    """MCP Protocol Endpoint exposed securely via local HTTP binding."""
    try:
        return db.search_similar_codes(payload.query_vector, payload.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP Search Tool Exception: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=config.MCP_QDRANT_SERVER_PORT)