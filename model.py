# ============================================================
# Hybrid Cross-Gated Fake News Detector
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFakeNewsDetector(nn.Module):

    def __init__(self):
        super().__init__()

        # -----------------------------
        # Text Projection
        # -----------------------------
        self.text_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # -----------------------------
        # Image Projection
        # -----------------------------
        self.image_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # -----------------------------
        # Cross Modal Attention
        # -----------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True,
            dropout=0.2
        )

        # -----------------------------
        # Cross Gating
        # -----------------------------
        self.text_gate = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )

        self.image_gate = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )

        # Learnable modality weights
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # -----------------------------
        # Fusion Layer
        # -----------------------------
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512)
        )

        # Residual Connection
        self.residual = nn.Linear(1024, 512)

        # -----------------------------
        # Classifier
        # -----------------------------
        self.classifier = nn.Sequential(

            nn.Linear(512,384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(384,192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(192,64),
            nn.GELU(),

            nn.Linear(64,2)
        )

    def forward(self, text, image):

        # Feature Projection
        text = self.text_proj(text)
        image = self.image_proj(image)

        # Normalize projected features
        text = F.normalize(text, p=2, dim=1)
        image = F.normalize(image, p=2, dim=1)

        # Add sequence dimension
        text = text.unsqueeze(1)
        image = image.unsqueeze(1)

        # Cross Attention
        attended, _ = self.cross_attention(
            text,
            image,
            image
        )

        attended = attended.squeeze(1)
        text = text.squeeze(1)
        image = image.squeeze(1)

        # Cross Gating
        gated_text = text * self.image_gate(image)
        gated_image = image * self.text_gate(text)

        # Learnable modality weighting
        gated_text = self.alpha * gated_text
        gated_image = self.beta * gated_image

        # Fusion
        fusion = torch.cat([gated_text, gated_image], dim=1)

        out = self.fusion(fusion)

        # Residual Connection
        out = out + self.residual(fusion)

        # Classification
        output = self.classifier(out)

        return output
    