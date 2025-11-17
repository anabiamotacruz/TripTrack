from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from services.embeddings import getTextEmbeddings, getImageEmbeddings, calculateSimilarity, calculateImageVibe
import numpy as np
import pandas as pd
import os
import base64
from io import BytesIO
from PIL import Image

router = APIRouter()

# Carregar o CSV de músicas
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tracks_with_clusters.csv")
tracks_df = pd.read_csv(CSV_PATH)

# Modelo para requisição de recomendação
class SongRecommendationRequest(BaseModel):
    query_text: Optional[str] = None
    query_image_base64: Optional[str] = None
    top_k: int = 5

# Modelo para músicas
class Song(BaseModel):
    artist: str
    track: str
    track_url: str
    tags: str
    description: str
    cluster: int
    vibe: str

@router.post("/")
async def getSongsRecommendation(request: SongRecommendationRequest):
    """
    Retorna recomendações de músicas baseadas em texto ou imagem usando embeddings CLIP
    
    Args:
        request: Objeto com query_text ou query_image_path e número de recomendações (top_k)
        
    Returns:
        Lista de músicas recomendadas com scores de similaridade e vibes da imagem (se aplicável)
    """
    try:
        # Validar entrada
        if not request.query_text and not request.query_image_base64:
            raise HTTPException(
                status_code=400, 
                detail="Forneça query_text ou query_image_base64"
            )
        
        # Inicializar variáveis
        image_vibes = None
        top_vibes = None
        recommendations = []
        
        # Processar query de imagem
        if request.query_image_base64:
            # Decodificar imagem base64
            try:
                image_data = base64.b64decode(request.query_image_base64)
                image = Image.open(BytesIO(image_data))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Erro ao decodificar imagem base64: {str(e)}"
                )
            
            # Calcular vibes da imagem
            image_vibes = calculateImageVibe(image)
            
            # Pegar as 5 vibes mais predominantes
            top_vibes = dict(list(image_vibes.items())[:5])
            
            # Pegar as 2 vibes mais predominantes para buscar músicas
            top_2_vibes = list(image_vibes.keys())[:2]
            
            # Buscar 5 músicas de cada vibe no CSV
            for vibe in top_2_vibes:
                # Filtrar músicas por vibe
                vibe_tracks = tracks_df[tracks_df['vibe'] == vibe].head(5)
                
                # Adicionar músicas às recomendações
                for _, track in vibe_tracks.iterrows():
                    recommendations.append({
                        "artist": track['artist'],
                        "track": track['track'],
                        "vibe_match": vibe,
                        "vibe_score": image_vibes[vibe]
                    })
        else:
            # Processar query de texto (busca por similaridade)
            query_embedding = getTextEmbeddings(request.query_text)
            
            # Calcular similaridade com cada música do CSV
            track_similarities = []
            for _, track in tracks_df.iterrows():
                # Gerar embedding da descrição da música
                song_embedding = getTextEmbeddings(track['description'])
                
                # Calcular similaridade
                similarity_score = calculateSimilarity(query_embedding, song_embedding)
                
                track_similarities.append({
                    "artist": track['artist'],
                    "track": track['track'],
                    "similarity_score": similarity_score
                })
            
            # Ordenar por similaridade (maior primeiro)
            track_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Pegar top_k recomendações
            recommendations = track_similarities[:request.top_k]
        
        # Preparar resposta
        response = {
            "query": {
                "text": request.query_text,
                "has_image": bool(request.query_image_base64)
            },
            "recommendations": recommendations,
            "total_found": len(recommendations)
        }
        
        # Adicionar vibes se a query foi uma imagem
        if top_vibes:
            response["top_vibes"] = top_vibes
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar recomendações: {str(e)}")


@router.get("/songs")
async def get_all_songs():
    """Retorna todas as músicas disponíveis no banco de dados"""
    songs = tracks_df.to_dict('records')
    return {"songs": songs, "total": len(songs)}


@router.get("/songs/vibe/{vibe_name}")
async def get_songs_by_vibe(vibe_name: str):
    """Retorna músicas filtradas por vibe"""
    filtered_tracks = tracks_df[tracks_df['vibe'] == vibe_name]
    if filtered_tracks.empty:
        raise HTTPException(status_code=404, detail=f"Nenhuma música encontrada para a vibe '{vibe_name}'")
    songs = filtered_tracks.to_dict('records')
    return {"vibe": vibe_name, "songs": songs, "total": len(songs)}


@router.get("/vibes")
async def get_all_vibes():
    """Retorna todas as vibes disponíveis no banco de dados"""
    vibes = tracks_df['vibe'].unique().tolist()
    vibe_counts = tracks_df['vibe'].value_counts().to_dict()
    return {"vibes": vibes, "counts": vibe_counts}
