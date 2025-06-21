from fastapi import FastAPI
from pydantic import BaseModel
from nomic.embed import EmbeddingClient
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "API de embeddings con Nomic funcionando correctamente"})

class Item(BaseModel):
    text: list[str]

@app.post("/embed")
async def get_embed(item: Item):
    client = EmbeddingClient(model='nomic-embed-text-v1.5')
    vectors = await client.embed(item.text)
    return {"vectors": vectors}





