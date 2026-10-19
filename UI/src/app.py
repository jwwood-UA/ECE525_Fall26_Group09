from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import sys
from pathlib import Path
folderPath = Path(__file__).resolve().parent
sys.path.append(str(folderPath.parent.parent / "backend"))
from logit_lens import GPT2LogitLens

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, never per request
    gpt2_model = GPT2LogitLens(
        model="gpt2",
        device="cpu",
        cache_dir=None,
        local_files_only=False,
    )
    app.state = {"gpt2": gpt2_model}
    yield

app = FastAPI(lifespan=lifespan)

class LensRequest(BaseModel):
    prompt: str = Field(max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    model: str = Field(max_length=10, default="gpt2")
    max_new_tokens: int = Field(default=1, ge=1, le=5)

@app.post("/api/logit_lens")
def logit_lens(req: LensRequest):
    return app.state[req.model].analyze(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        model=req.model,
        top_k=req.top_k,
    )
    

app.mount("/", StaticFiles(directory=str(folderPath / "static"),html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8767, reload=True)