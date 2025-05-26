from fastapi import FastAPI

app = FastAPI(title="API Agent")

@app.get("/health")
def health():
    return {"status": "ok", "agent": "API Agent"}
