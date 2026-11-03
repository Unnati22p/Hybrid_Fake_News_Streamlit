# 🧠 Hybrid Explainable Fake News Detection

### A Multimodal Deep Learning Framework for Detecting Fake News on Social Media

An AI-powered fake news detection system integrating Natural Language Processing (NLP), Computer Vision, and Explainable AI (XAI) to analyse textual and visual content from social media posts.

The framework combines transformer-based text representations and image embeddings through an attention-based fusion architecture to improve multimodal misinformation detection.

---

## 🎯 Project Overview

Fake news spreads rapidly across social media platforms, often combining misleading text with manipulated or unrelated images.

Traditional text-only detection approaches may overlook visual misinformation.

This project explores a hybrid multimodal deep learning approach that analyses both textual and visual information to identify potentially misleading social media content.

## ✨ Key Features

- **Multimodal Analysis:** Integrates textual and visual information.
- **Text Representation:** Uses RoBERTa-based contextual embeddings.
- **Image Representation:** Uses OpenCLIP for visual feature extraction.
- **Attention-Based Fusion:** Combines text and image representations.
- **Deep Learning:** Implements a custom hybrid neural network.
- **Explainable AI:** Explores explainability for model predictions.
- **Interactive Interface:** Streamlit-based application for predictions.

## 🏗️ Model Architecture

```text
                 Social Media Post
                        |
              +---------+---------+
              |                   |
            Text                 Image
              |                   |
           RoBERTa              OpenCLIP
              |                   |
        Text Features        Image Features
              |                   |
        Text Projection      Image Projection
              |                   |
              +---------+---------+
                        |
                Attention-Based
                     Fusion
                        |
                Classification
                    Network
                        |
                Fake / Real
                  Prediction
```

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| NLP | RoBERTa, Transformers |
| Computer Vision | OpenCLIP, ViT |
| Multimodal Learning | Attention Mechanisms |
| Explainable AI | XAI |
| Interface | Streamlit |
| Development | Google Colab, GitHub |

## 📂 Project Structure

```text
Hybrid_Fake_News_Streamlit/
│
├── .devcontainer/
├── checkpoint/
├── .gitignore
├── app.py
├── inference.py
├── model.py
├── requirements.txt
├── runtime.txt
└── README.md
```
## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Unnati22p/Hybrid_Fake_News_Streamlit.git
```

### 2. Navigate to Project
```bash
cd Hybrid_Fake_News_Streamlit
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run app.py
```
## 📊 Research Focus

This project explores:

- Multimodal fake news detection.
- Contextual text feature extraction.
- Visual representation learning.
- Attention-based feature fusion.
- Explainable deep learning.
- Performance evaluation against existing approaches.

## 🔬 Research Background

Developed as part of research internship work at **Indian Institute of Information Technology Pune (IIIT Pune)**.

**Research Area:** Hybrid Explainable Deep Learning Framework for Fake News Detection on Social Media.

## ⚠️ Disclaimer

This project is a research-oriented misinformation detection system.

Model predictions should not be treated as definitive verification of factual accuracy.

## 👩‍💻 Author

**Unnati Patil**

Computer Science Engineering | Data Science | AI & ML

- GitHub: https://github.com/Unnati22p
- LinkedIn: https://www.linkedin.com/in/unnati-patil2277

---
⭐ If you find this project interesting, consider starring the repository.
