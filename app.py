from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import nomic

# Login con tu API Key
nomic.login(os.environ["NOMIC_API_KEY"])

from nomic import embed

app = FastAPI()

# Modelo de entrada
class Item(BaseModel):
    text: List[str]

@app.post("/embed")
async def get_embed(item: Item):
    vectors = embed.text(
        texts=item.text,
        model="nomic-embed-text-v1",
        task_type="search_document",     # ✅ task_type válido
        dimensionality=256               # ✅ (aunque se ignora en v1, no da error)
    )["embeddings"]                      # ✅ Extraer solo los vectores

    return {"vectors": vectors}











