from fastapi import FastAPI
from pydantic import BaseModel
from nomic import embed
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "API de embeddings con Nomic funcionando correctamente"})

class Item(BaseModel):
    text: list[str]

@app.post("/embed")
async def get_embed(item: Item):
    # Usa embed.text para modelos nomic-embed-text-v1.5
    vectors = embed.text(
        texts=item.text,
        model="nomic-embed-text-v1.5",
        task_type="search_document",
        dimensionality=512  # o 256 u 768 según lo que necesites
    )
    return JSONResponse(content={"vectors": vectors})






