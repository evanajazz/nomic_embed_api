from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from nomic_embed import embed  # ← Aquí es el import correcto

app = FastAPI()

# Verificación
@app.get("/")
def read_root():
    return JSONResponse(content={"message": "API de embeddings con Nomic funcionando correctamente"})

# Entrada de datos
class Item(BaseModel):
    text: list[str]

# Ruta para generar embeddings
@app.post("/embed")
async def get_embed(item: Item):
    vectors = await embed(item.text)
    return {"vectors": vectors}




