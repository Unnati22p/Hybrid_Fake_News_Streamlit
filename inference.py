# ============================================================
# inference.py
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import os

from PIL import Image

import open_clip

from transformers import (
    RobertaTokenizer,
    RobertaModel
)

from model import HybridFakeNewsDetector

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Load RoBERTa
# ------------------------------------------------------------

tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base"
)

text_encoder = RobertaModel.from_pretrained(
    "roberta-base"
).to(DEVICE)

text_encoder.eval()

# ------------------------------------------------------------
# Load OpenCLIP
# ------------------------------------------------------------

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

clip_model = clip_model.to(DEVICE)

clip_model.eval()

# ------------------------------------------------------------
# Load Hybrid Model
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoint",
    "final_cross_gated_model.pth"
)

model = HybridFakeNewsDetector()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)

model.eval()

CLASS_NAMES = [
    "Fake News",
    "Real News"
]
# ============================================================
# Text Feature Extraction
# ============================================================

@torch.no_grad()
def extract_text_feature(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    outputs = text_encoder(**inputs)

    feature = outputs.last_hidden_state[:, 0, :]

    feature = F.normalize(
        feature,
        p=2,
        dim=1
    )

    return feature


# ============================================================
# Image Feature Extraction
# ============================================================

@torch.no_grad()
def extract_image_feature(image):

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    else:
        image = image.convert("RGB")

    image = preprocess(image)

    image = image.unsqueeze(0).to(DEVICE)

    feature = clip_model.encode_image(image)

    feature = F.normalize(
        feature,
        p=2,
        dim=1
    )

    return feature


# ============================================================
# Gate Score Extraction
# ============================================================

@torch.no_grad()
def get_gate_scores(text_feature, image_feature):

    text = model.text_proj(text_feature)

    image = model.image_proj(image_feature)

    text_gate = model.image_gate(image)

    image_gate = model.text_gate(text)

    text_score = float(
        text_gate.mean().cpu()
    )

    image_score = float(
        image_gate.mean().cpu()
    )

    return text_score, image_score

# ============================================================
# Prediction Engine
# ============================================================

@torch.no_grad()
def predict(text, image):

    # --------------------------------------------------------
    # Extract Features
    # --------------------------------------------------------

    text_feature = extract_text_feature(text)

    image_feature = extract_image_feature(image)

    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    logits = model(
        text_feature,
        image_feature
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

    prediction = prediction.item()

    confidence = confidence.item()

    probabilities = probabilities.squeeze().cpu().numpy()

    # --------------------------------------------------------
    # Gate Scores
    # --------------------------------------------------------

    text_gate, image_gate = get_gate_scores(
        text_feature,
        image_feature
    )

    # --------------------------------------------------------
    # Contribution Percentages
    # --------------------------------------------------------

    total = text_gate + image_gate

    text_percent = (text_gate / total) * 100

    image_percent = (image_gate / total) * 100

    # --------------------------------------------------------
    # Result Dictionary
    # --------------------------------------------------------

    result = {

        "prediction": CLASS_NAMES[prediction],

        "confidence": confidence,

        "fake_probability": float(probabilities[0]),

        "real_probability": float(probabilities[1]),

        "text_gate": text_gate,

        "image_gate": image_gate,

        "text_contribution": text_percent,

        "image_contribution": image_percent

    }

    return result

# ============================================================
# Human Readable Explanation
# ============================================================

def generate_explanation(result):

    prediction = result["prediction"]

    confidence = result["confidence"] * 100

    text_percent = result["text_contribution"]

    image_percent = result["image_contribution"]

    if text_percent > image_percent:
        dominant = "textual information"
    else:
        dominant = "visual information"

    explanation = (
        f"The model classified the news as {prediction} "
        f"with a confidence of {confidence:.2f}%. "
        f"It relied more on {dominant} while making the prediction. "
        f"Estimated contribution: "
        f"Text = {text_percent:.2f}%, "
        f"Image = {image_percent:.2f}%."
    )

    return explanation


# ============================================================
# Complete Inference Pipeline
# ============================================================

def run_inference(text, image):

    result = predict(text, image)

    result["explanation"] = generate_explanation(result)

    return result


# ============================================================
# Optional Local Test
# (This runs only if inference.py is executed directly)
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Inference module loaded successfully.")
    print("This file is intended to be imported by app.py.")
    print("=" * 60)