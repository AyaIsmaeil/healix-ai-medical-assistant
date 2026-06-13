from fastapi import FastAPI
from app.routes.chat import router as chat_router

app = FastAPI(
    title="Healix AI Medical Assistant",
    version="1.0"
)

app.include_router(chat_router, prefix="/chat")


@app.get("/")
def home():
    return {"message": "Healix is running"}