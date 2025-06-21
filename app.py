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
    vectors = embed.text(
        texts=item.text,
        model="nomic-embed-text-v1.5",
        task_type="search_document",
        dimensionality=256
    )
    return {"vectors": vectors}







