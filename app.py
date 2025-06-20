from fastapi import FastAPI
from pydantic import BaseModel
from nomic import embed
from fastapi.responses import JSONResponse

app = FastAPI()

# Ruta raíz para verificar que la API está viva
@app.get("/")
def read_root():
    return JSONResponse(content={"message": "API de embeddings con Nomic funcionando correctamente"})

# Modelo para la entrada de texto
class Item(BaseModel):
    text: list[str]

# Ruta para generar embeddings
@app.post("/embed")
async def get_embed(item: Item):
    vectors = [v async for v in embed(item.text)]
    return {"vectors": vectors}

