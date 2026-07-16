import os
from io import BytesIO
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import AutoModel, AutoProcessor

app = FastAPI(
    title="StyLoop Outfit Recommendation ML Microservice",
    description="Standalone FastAPI microservice running Marqo/marqo-fashionSigLIP to generate outfit recommendations",
    version="1.0"
)

# Global variables for model, processor, and wardrobe
model = None
processor = None
wardrobe_embeddings = []
wardrobe_metadata = []

# List of dummy wardrobe items
DUMMY_WARDROBE = [
    {"item_id": "item_1", "name": "Classic White T-Shirt", "path": "white_tshirt.jpg"},
    {"item_id": "item_2", "name": "Slim Fit Blue Jeans", "path": "blue_jeans.jpg"},
    {"item_id": "item_3", "name": "Black Leather Jacket", "path": "black_jacket.jpg"},
    {"item_id": "item_4", "name": "Red Running Shoes", "path": "red_shoes.jpg"},
    {"item_id": "item_5", "name": "Casual Brown Belt", "path": "brown_belt.jpg"},
]

@app.on_event("startup")
def startup_event():
    global model, processor, wardrobe_embeddings, wardrobe_metadata
    print("Loading Marqo/marqo-fashionSigLIP model and processor...")
    
    # Load model & processor from Hugging Face
    model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
    processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
    model.eval()
    
    # Run a dummy forward pass to get the embedding dimension
    dummy_img = Image.new("RGB", (224, 224), color="white")
    inputs = processor(images=dummy_img, return_tensors="pt")
    with torch.no_grad():
        dummy_embedding = model.get_image_features(inputs['pixel_values'], normalize=True)
    embedding_dim = dummy_embedding.shape[-1]
    print(f"Model loaded successfully. Embedding dimension: {embedding_dim}")
    
    # Encode wardrobe items
    encoded_embeddings = []
    for item in DUMMY_WARDROBE:
        path = item["path"]
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    emb = model.get_image_features(inputs['pixel_values'], normalize=True)
                emb_np = emb.cpu().numpy()[0]
                # L2 normalize
                emb_np = emb_np / np.linalg.norm(emb_np)
                encoded_embeddings.append(emb_np)
                wardrobe_metadata.append(item)
                print(f"Loaded and encoded: {path}")
            except Exception as e:
                print(f"Error encoding {path}: {e}")
        else:
            # Fallback: Generate dummy embedding if file doesn't exist
            rand_emb = np.random.randn(embedding_dim).astype(np.float32)
            rand_emb = rand_emb / np.linalg.norm(rand_emb)
            encoded_embeddings.append(rand_emb)
            wardrobe_metadata.append(item)
            print(f"Image not found. Initialized dummy embedding for: {item['name']} ({path})")
            
    # Convert list of embeddings to numpy array
    wardrobe_embeddings = np.array(encoded_embeddings)

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "StyLoop Outfit Recommendation ML Microservice",
        "model": "Marqo/marqo-fashionSigLIP",
        "cached_items": len(wardrobe_metadata)
    }

@app.post("/recommend")
async def recommend(image: UploadFile = File(...)):
    global model, processor, wardrobe_embeddings, wardrobe_metadata
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is loading, please try again in a few seconds.")
        
    try:
        # Read uploaded image bytes
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Process and extract features
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(inputs['pixel_values'], normalize=True)
        query_emb = emb.cpu().numpy()[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Calculate cosine similarities using dot product
        similarities = wardrobe_embeddings @ query_emb
        
        # Pair metadata with similarity scores
        results = []
        for i, item in enumerate(wardrobe_metadata):
            results.append({
                "item_id": item["item_id"],
                "name": item["name"],
                "similarity_score": round(float(similarities[i]), 4)
            })
            
        # Sort by similarity score descending
        results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "success": True,
            "recommendations": results[:5]
        }
        
    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
