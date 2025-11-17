from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import recommendations

app = FastAPI(
    title="TripTrack API",
    description="API para recomendação de músicas usando CLIP",
    version="1.0.0"
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rotas
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["Recommendations"])

@app.get("/")
async def root():
    return {
        "message": "Bem-vindo ao TripTrack API - Sistema de Recomendação de Músicas",
        "endpoints": {
            "recommendations": "/api/recommendations",
            "songs": "/api/recommendations/songs",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "CLIP-ViT-B/32"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
