from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, never per request
    yield

app = FastAPI(lifespan=lifespan)

class LensRequest(BaseModel):
    prompt: str = Field(max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

# TODO: trigger logit lens run with prompt and return json data
# TODO: define json return format
@app.post("/api/logit_lens")
def logit_lens(req: LensRequest):
    return {}

app.mount("/", StaticFiles(directory="static",html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8767, reload=True)