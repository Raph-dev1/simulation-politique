import os
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

# 1. Désactiver le parallélisme pour économiser la RAM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Utilisation d'un modèle plus petit et optimisé
# 'paraphrase-multilingual-MiniLM-L12-v2' est 4x plus léger que CamemBERT
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Chargement du modèle optimisé...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# low_cpu_mem_usage=True permet d'éviter les pics de RAM au chargement
model = AutoModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
model.eval()

class SimulationRequest(BaseModel):
    profiles: List[str]
    programs: Dict[str, str]

@torch.no_grad()
def get_embeddings(texts: List[str]):
    # On traite par petits paquets (batchs) pour ne pas saturer la RAM
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    # Mean Pooling
    mask = inputs["attention_mask"].unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return embeddings.numpy()

@app.post("/simulate")
async def simulate(req: SimulationRequest):
    try:
        party_names = list(req.programs.keys())
        program_texts = list(req.programs.values())
        
        # Calcul des vecteurs
        prog_vecs = get_embeddings(program_texts)
        prof_vecs = get_embeddings(req.profiles)
        
        # Similarité cosinus
        scores = np.dot(prof_vecs, prog_vecs.T)
        
        return {"status": "success", "labels": party_names, "scores": scores.tolist()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
