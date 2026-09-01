from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import forecast, chat

app = FastAPI(
    title="BITS ERP AI Service",
    description="Forecasting and AI chat for BITS ERP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router)
app.include_router(chat.router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "bitserp-ai"}