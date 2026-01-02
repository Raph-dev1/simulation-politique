import os
import uvicorn
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from transformers import CamembertTokenizer, CamembertModel

# 1. Initialisation de l'API
app = FastAPI(title="Moteur de Simulation Politique")

# 2. Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Chargement du modèle CamemBERT (CPU)
DEVICE = "cpu"
MODEL_NAME = "camembert-base"

print("Chargement de CamemBERT... (cela peut prendre quelques minutes)")
tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
bert = CamembertModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert.eval()
print("Modèle chargé avec succès.")

class SimulationRequest(BaseModel):
    profiles: List[str]
    programs: Dict[str, str]

@torch.no_grad()
def get_embeddings(texts: List[str]):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = bert(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(embeddings, p=2, dim=1).cpu().numpy()

@app.post("/simulate")
async def simulate(req: SimulationRequest):
    try:
        party_names = list(req.programs.keys())
        program_texts = list(req.programs.values())
        program_vecs = get_embeddings(program_texts)
        profile_vecs = get_embeddings(req.profiles)
        scores = np.dot(profile_vecs, program_vecs.T)

        return {
            "status": "success",
            "labels": party_names,
            "scores": scores.tolist()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
