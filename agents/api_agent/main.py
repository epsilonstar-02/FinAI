# Sets up FastAPI app, includes routes, exception handlers, and docs.

from fastapi import FastAPI, HTTPException, Depends
from .models import PriceRequest, PriceResponse, HistoricalRequest, HistoricalResponse
from .client import fetch_current_price, fetch_historical, APIClientError
from .config import settings
import uvicorn

app = FastAPI(title="API Agent", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "agent": "API Agent"}


@app.get("/price", response_model=PriceResponse)
async def get_price(request: PriceRequest = Depends()):
    try:
        data = await fetch_current_price(request.symbol)
    except APIClientError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return data


@app.get("/historical", response_model=HistoricalResponse)
async def get_historical(request: HistoricalRequest = Depends()):
    if request.start > request.end:
        raise HTTPException(
            status_code=400, detail="start date must be before end date"
        )
    try:
        data = await fetch_historical(request.symbol, request.start, request.end)
    except APIClientError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return data


if __name__ == "__main__":
    uvicorn.run("agents.api_agent.main:app", host="0.0.0.0", port=8001, reload=True)
