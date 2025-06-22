from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import nomic

nomic.login(os.environ["NOMIC_API_KEY"])  # ✅ Login correcto

from nomic import embed

app = FastAPI()

class Item(BaseModel):
    text: List[str]

@app.post("/embed")
async def get_embed(item: Item):
    vectors = embed.text(
        texts=item.text,
        model="nomic-embed-text-v1",  # ✅ Modelo correcto
        task_type="retrieval",        # ✅ "retrieval" es el único válido actualmente
        dimensionality=256            # ✅ Dimensionalidad aceptada
    )
    return {"vectors": vectors}










