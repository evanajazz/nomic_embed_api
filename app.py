from fastapi import FastAPI
from pydantic import BaseModel
from nomic import EmbeddingClient
from fastapi.responses import JSONResponse

app = FastAPI()

# Ruta raíz
@app.get("/")
def read_root():
    return JSONResponse(content={"message": "API de embeddings con Nomic funcionando correctamente"})

# Modelo de entrada
class Item(BaseModel):
    text: list[str]

# Ruta para generar embeddings
@app.post("/embed")
async def get_embed(item: Item):
    client = EmbeddingClient(model="nomic-embed-text-v1")  # puedes cambiar el modelo si lo deseas
    vectors = client.embed(item.text)
    return {"vectors": vectors}


