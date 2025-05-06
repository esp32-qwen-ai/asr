from fastapi import FastAPI, Response
from pydantic import BaseModel
from asr import ASR
import base64

asr = ASR()

app = FastAPI()

class ASRRequest(BaseModel):
    pcm: str
    sample_rate: int = 16000
    is_finish: bool = False

@app.post("/asr")
async def asr_endpoint(request: ASRRequest):
    pcm_bytes = base64.b64decode(request.pcm)
    result = asr.call(pcm_bytes, request.is_finish)
    return Response(content=result, media_type="text/plain")

@app.get("/ping")
async def ping():
    return "pong"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)