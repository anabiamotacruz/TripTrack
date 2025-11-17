import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Union, List, Dict

# Carregar modelo CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Definir vibes e prompts
VIBES_LABELS = [
    "Confident",
    "Determined",
    "Nostalgic",
    "Reflective",
    "Euphoric",
    "Energetic",
    "Calm",
    "Melancholic",
]

VIBE_PROMPT = [
    f"A photograph that evokes a deep sense of {vibe.lower()}, capturing the mood and atmosphere associated with it." 
    for vibe in VIBES_LABELS
]

def getTextEmbeddings(text: Union[str, List[str]]) -> np.ndarray:
    """
    Gera embeddings de texto usando CLIP
    
    Args:
        text: String ou lista de strings para gerar embeddings
        
    Returns:
        np.ndarray: Array numpy com os embeddings normalizados
    """
    if isinstance(text, str):
        text = [text]
    
    # Processar texto
    inputs = processor(text=text, return_tensors="pt", padding=True)
    
    # Gerar embeddings
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    # Normalizar embeddings
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy()


def getImageEmbeddings(image: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> np.ndarray:
    """
    Gera embeddings de imagem usando CLIP
    
    Args:
        image: Caminho da imagem, objeto PIL Image ou lista deles
        
    Returns:
        np.ndarray: Array numpy com os embeddings normalizados
    """
    # Converter para lista se não for
    if not isinstance(image, list):
        image = [image]
    
    # Carregar imagens se forem caminhos
    images = []
    for img in image:
        if isinstance(img, str):
            images.append(Image.open(img))
        else:
            images.append(img)
    
    # Processar imagens
    inputs = processor(images=images, return_tensors="pt", padding=True)
    
    # Gerar embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Normalizar embeddings
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()


def calculateSimilarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula a similaridade de cosseno entre dois embeddings
    
    Args:
        embedding1: Primeiro embedding (numpy array)
        embedding2: Segundo embedding (numpy array)
        
    Returns:
        float: Similaridade de cosseno (0 a 1)
    """
    # Se os embeddings têm múltiplas dimensões, pegar a primeira
    if len(embedding1.shape) > 1:
        embedding1 = embedding1[0]
    if len(embedding2.shape) > 1:
        embedding2 = embedding2[0]
    
    # Calcular produto escalar (já que embeddings estão normalizados)
    similarity = np.dot(embedding1, embedding2)
    
    # Converter para float Python e garantir que está entre 0 e 1
    return float(max(0.0, min(1.0, similarity)))


def calculateImageVibe(image: Union[str, Image.Image]) -> Dict[str, float]:
    """
    Calcula a "vibe" de uma imagem comparando com descrições de diferentes vibes
    
    Args:
        image: Caminho da imagem ou objeto PIL Image
        
    Returns:
        Dict[str, float]: Dicionário com cada vibe e seu score de similaridade,
                         ordenado do maior para o menor score
    """
    # Gerar embedding da imagem
    image_embedding = getImageEmbeddings(image)
    
    # Gerar embeddings de todos os prompts de vibe
    vibe_embeddings = getTextEmbeddings(VIBE_PROMPT)
    
    # Calcular similaridade para cada vibe
    vibe_scores = {}
    for i, vibe_label in enumerate(VIBES_LABELS):
        vibe_embedding = vibe_embeddings[i:i+1]  # Manter dimensão
        similarity = calculateSimilarity(image_embedding, vibe_embedding)
        vibe_scores[vibe_label] = similarity
    
    # Ordenar por score (maior primeiro)
    sorted_vibes = dict(sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_vibes

