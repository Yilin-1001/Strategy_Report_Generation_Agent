"""
BGE-M3 Embedding Server for LangFlow Integration
This server provides HTTP API endpoints for BGE-M3 embeddings
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
import os

app = FastAPI(title="BGE-M3 Embedding Server")

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
DEVICE = os.getenv("DEVICE", "cuda")
CACHE_DIR = os.getenv("CACHE_DIR", "./data/models")
PORT = int(os.getenv("PORT", 8080))

print(f"Loading model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Cache dir: {CACHE_DIR}")

# Load model on startup
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
    cache_folder=CACHE_DIR,
    trust_remote_code=True
)

print(f"Model loaded successfully! Dimension: {model.get_sentence_embedding_dimension()}")


class EmbeddingRequest(BaseModel):
    """Request model for single text embedding"""
    input: str
    model: str = "bge-m3"


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding"""
    inputs: List[str]
    model: str = "bge-m3"


class EmbeddingResponse(BaseModel):
    """Response model for embeddings"""
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "BGE-M3 Embedding Server",
        "model": MODEL_NAME,
        "dimension": model.get_sentence_embedding_dimension(),
        "device": DEVICE
    }


@app.get("/health")
async def health():
    """Health check for LangFlow"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "dimension": model.get_sentence_embedding_dimension()
    }


@app.post("/embeddings")
async def create_embeddings(request: Union[EmbeddingRequest, BatchEmbeddingRequest]):
    """
    Create embeddings for text(s)
    Compatible with OpenAI embedding API format
    """
    try:
        # Handle single or batch input
        if isinstance(request, EmbeddingRequest):
            texts = [request.input]
        else:
            texts = request.inputs

        # Generate embeddings
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Format response (OpenAI-compatible)
        embedding_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        data = []
        for i, emb in enumerate(embedding_list):
            data.append({
                "object": "embedding",
                "embedding": emb,
                "index": i
            })

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def create_embeddings_v1(request: Union[EmbeddingRequest, BatchEmbeddingRequest]):
    """
    OpenAI v1 compatible endpoint
    """
    return await create_embeddings(request)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
