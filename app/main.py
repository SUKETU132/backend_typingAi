from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router
from app.routes.upload import router as upload_router
from app.routes.results import router as results_router
from app.routes.auth import router as auth_router

app = FastAPI(title="TypingAI API", version="2.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(auth_router,    prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.include_router(upload_router,  prefix="/api/v1")
app.include_router(results_router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "TypingAI API v2 — Running"}