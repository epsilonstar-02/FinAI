import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default configuration
host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8001"))

if __name__ == "__main__":
    print(f"Starting Retriever Agent on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
