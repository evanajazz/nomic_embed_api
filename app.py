from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from nomic.embeddings import embed

app = FastAPI()

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "API de embeddings con Nomic funcionando correctamente"})

class Item(BaseModel):
    text: list[str]

@app.post("/embed")
async def get_embed(item: Item):
    vectors = embed(texts=item.text, model='nomic-embed-text-v1', task_type='embedding')
    return {"vectors": vectors}



