from fastapi import FastAPI
from pydantic import BaseModel
from nomic import embed

app = FastAPI()

class Item(BaseModel):
    text: list[str]

@app.post("/embed")
async def get_embed(item: Item):
    vectors = embed(item.text)
    return {"vectors": vectors}
