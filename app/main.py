import os

import torch
from fastapi import FastAPI
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from app.model import LungCancerNet


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def download_model(filename: str):

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set.")

    if not HF_REPO_ID:
        raise RuntimeError("HF_REPO_ID is not set.")

    print(
        f"Downloading {filename} from "
        f"Hugging Face repository {HF_REPO_ID}..."
    )

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=HF_TOKEN
    )

    print(f"Downloaded {filename}")

    return path


def load_lung_cancer_model():

    checkpoint_path = download_model(
        "Proposed_LungCancerNet_best.pth"
    )

    model = LungCancerNet(
        num_classes=4,
        freeze_backbones=False
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint,
        strict=False
    )

    model.to(DEVICE)
    model.eval()

    return model


app = FastAPI(
    title="Lung Cancer API"
)


# Load model when API starts
model = None

try:
    model = load_lung_cancer_model()
    print("Model loaded successfully.")

except Exception as e:
    print(f"Model loading failed: {e}")


# ============================================================
# GET ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Lung Cancer API is running"
    }


# ============================================================
# MODEL STATUS ROUTE
# ============================================================

@app.get("/health")
def health():

    return {
        "model_loaded": model is not None
    }