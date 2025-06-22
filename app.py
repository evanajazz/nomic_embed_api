from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import nomic

nomic.login(os.environ["NOMIC_API_KEY"])

from nomic import embed

app = FastAPI()

class Item(BaseModel):
    text: List[str]

@app.post("/embed")
async def get_embed(item: Item):
    vectors = embed.text(
        texts=item.text,
        model="nomic-embed-text-v1",
        task_type="retrieval",  # ✅ Cambiado de "embedding"
        dimensionality=256
    )
    return {"vectors": vectors}









