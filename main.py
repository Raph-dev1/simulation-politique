{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww29200\viewh18380\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import uvicorn\
import numpy as np\
import torch\
import torch.nn.functional as F\
from fastapi import FastAPI\
from fastapi.middleware.cors import CORSMiddleware\
from pydantic import BaseModel\
from typing import List, Dict\
from transformers import CamembertTokenizer, CamembertModel\
\
# 1. Initialisation de l'API\
app = FastAPI(title="Moteur de Simulation Politique")\
\
# 2. Configuration CORS pour Lovable\
# Cela permet \'e0 ton site Lovable d'appeler ce serveur sans blocage de s\'e9curit\'e9\
app.add_middleware(\
    CORSMiddleware,\
    allow_origins=["*"],\
    allow_methods=["*"],\
    allow_headers=["*"],\
)\
\
# 3. Chargement du mod\'e8le CamemBERT (version CPU pour Render Free)\
DEVICE = "cpu"\
MODEL_NAME = "camembert-base"\
\
print("Chargement de CamemBERT... (cela peut prendre quelques minutes)")\
tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)\
bert = CamembertModel.from_pretrained(MODEL_NAME).to(DEVICE)\
bert.eval()\
print("Mod\'e8le charg\'e9 avec succ\'e8s.")\
\
# 4. Mod\'e8les de donn\'e9es pour l'API\
class SimulationRequest(BaseModel):\
    profiles: List[str]  # Liste des 150 textes de profils \'e9lecteurs\
    programs: Dict[str, str] # \{"Gauche": "texte...", "Centre": "texte...", "Droite": "texte..."\}\
\
# 5. Logique de calcul s\'e9mantique\
@torch.no_grad()\
def get_embeddings(texts: List[str]):\
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)\
    outputs = bert(**inputs)\
    # On utilise la moyenne des \'e9tats cach\'e9s pour une meilleure pr\'e9cision s\'e9mantique\
    mask = inputs["attention_mask"].unsqueeze(-1)\
    embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)\
    return F.normalize(embeddings, p=2, dim=1).cpu().numpy()\
\
# 6. Endpoint principal pour la simulation\
@app.post("/simulate")\
async def simulate(req: SimulationRequest):\
    try:\
        # Calcul des vecteurs pour les programmes\
        party_names = list(req.programs.keys())\
        program_texts = list(req.programs.values())\
        program_vecs = get_embeddings(program_texts) # Shape: (Nb_Partis, 768)\
        \
        # Calcul des vecteurs pour les 150 profils\
        profile_vecs = get_embeddings(req.profiles) # Shape: (150, 768)\
        \
        # Calcul du matching (Produit scalaire = Similarit\'e9 cosinus ici car vecteurs normalis\'e9s)\
        # scores[i][j] = similarit\'e9 entre profil i et programme j\
        scores = np.dot(profile_vecs, program_vecs.T)\
        \
        return \{\
            "status": "success",\
            "labels": party_names,\
            "scores": scores.tolist()\
        \}\
    except Exception as e:\
        return \{"status": "error", "message": str(e)\}\
\
# 7. Lancement du serveur\
if __name__ == "__main__":\
    port = int(os.environ.get("PORT", 8000))\
    uvicorn.run(app, host="0.0.0.0", port=port)}