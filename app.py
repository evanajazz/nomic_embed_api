from fastapi import FastAPI
from pydantic import BaseModel
import os
import nomic
from nomic import embed

# Login explícito
nomic.login(api_key=os.environ["NOMIC_API_KEY"])

app = FastAPI()

class EmbedText(BaseModel):
    text: list

@app.post("/embed")
async def get_embed(item: EmbedText):
    vectors = embed.text(
        texts=item.text,
        model='nomic-embed-text-v1',
        task_type='embedding',
        dimensionality=256
    )
    return {"embeddings": vectors}








