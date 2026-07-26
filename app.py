import streamlit as st
from PIL import Image

from inference import run_inference

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="📰 Hybrid Explainable Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.title("📰 Hybrid Explainable Fake News Detection")

st.caption("An Explainable Multimodal Deep Learning Framework for Fake News Detection using RoBERTa, OpenCLIP and Cross-Gated Fusion.")

st.markdown("""

### Description

This application presents our proposed **Hybrid Cross-Gated Explainable Deep Learning Framework**
for multimodal fake news detection.
The framework simultaneously analyzes:
- 📝 News Text
- 🖼️ Associated News Image

to determine whether the news is **Real** or **Fake**.
Unlike traditional fake news detectors, our proposed model also provides explainable predictions,
including confidence scores, prediction probabilities, and the contribution of textual and visual
modalities toward the final decision.
""")

st.subheader("✨ Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ✅ Multimodal Fake News Detection

    ✅ RoBERTa Text Encoder

    ✅ OpenCLIP Image Encoder

    ✅ Cross-Gated Fusion
    """)

with col2:
    st.markdown("""
    ✅ Explainable AI (XAI)

    ✅ Confidence Score

    ✅ Prediction Probabilities

    ✅ Text & Image Contribution Analysis
    """)

st.divider()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Detection", "📖 About"]
)
# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":


    st.info(
        "Navigate to the Detection page from the sidebar to classify news."
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "📖 About":

    st.header("📖 About the Project")

    st.markdown("""

## Hybrid Explainable Fake News Detection
                
This project presents a **Hybrid Cross-Gated Explainable Deep Learning Framework**
for detecting fake news by jointly analyzing textual and visual information.
The proposed framework combines state-of-the-art language and vision models with
Cross-Gated Fusion to improve multimodal fake news detection while providing
human-understandable explanations.

""")
    
    st.divider()

    st.subheader("🎯 Objectives")

    st.markdown("""

    - Detect fake news using text and images
    - Improve multimodal classification accuracy
    - Provide explainable predictions
    - Estimate text and image contribution
    - Build an interactive Streamlit application

    """)

    st.divider()

    st.subheader("🛠 Technologies Used")

    st.markdown("""

     Python
    - PyTorch
    - RoBERTa
    - OpenCLIP
    - Streamlit
    - PIL
    - NumPy

    """)

    st.divider()

    st.subheader("🧠 Proposed Model")

    st.markdown("""

    The proposed architecture consists of:

    - RoBERTa Text Encoder
    - OpenCLIP Image Encoder
    - Cross-Gated Fusion Module
    - Classification Layer
    - Explainability Module

    """)

    st.divider()

    st.subheader("📊 Output")

    st.markdown("""

    The application provides:

    - Prediction (Real/Fake)
    - Confidence Score
    - Prediction Probabilities
    - Explainability
    - Text Contribution
    - Image Contribution
    - Gate Scores

    """)

    st.divider()

    st.info(

    "Developed as part of Research Internship on Explainable Multimodal Fake News Detection."

)
# ============================================================
# DETECTION PAGE
# ============================================================

elif page == "🔍 Detection":

    st.header("🔍 Fake News Detection")

    col1, col2 = st.columns([1, 1])

    with col1:

        uploaded_image = st.file_uploader(
            "Upload News Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:

            image = Image.open(uploaded_image).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_column_width=True
            )

        else:

            image = None

    with col2:

        news_text = st.text_area(
            "Enter News Text",
            height=250,
            placeholder="Paste the news article or headline here..."
        )

    st.divider()

    predict = st.button(
        "🚀 Detect Fake News",
        use_container_width=True
    )

    if predict:

        if image is None:

            st.error("Please upload an image.")

        elif news_text.strip() == "":

            st.error("Please enter the news text.")

        else:

            with st.spinner("Analyzing multimodal news..."):

                result = run_inference(
                    news_text,
                    image
                )

            st.success("Prediction Completed!")

            st.divider()

            prediction = result["prediction"]

            confidence = result["confidence"] * 100

            st.subheader("📋 Prediction Summary")

            if confidence >= 90:

                level = "Very High"

            elif confidence >= 80:

                level = "High"

            elif confidence >= 70:

                level = "Moderate"

            else:

                level = "Low"

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric("Prediction", prediction)

            with col2:

                st.metric("Confidence", f"{confidence:.2f}%")

            with col3:

                st.metric("Reliability", level)

            

            if prediction == "Real News":

                st.success(
                    f"### ✅ Prediction : {prediction}"
                )

            else:

                st.error(
                    f"### ❌ Prediction : {prediction}"
                )

            st.divider()

            # ============================================================
            # Probabilities
            # ============================================================

            st.subheader("📊 Prediction Probabilities")

            st.write(
                f"**Real News:** {result['real_probability']*100:.2f}%"
            )
            st.progress(float(result["real_probability"]))

            st.write(
                f"**Fake News:** {result['fake_probability']*100:.2f}%"
            )
            st.progress(float(result["fake_probability"]))

            st.divider()

            # ============================================================
            # Explainability
            # ============================================================

            st.subheader("🧠 Explainability")

            st.info(result["explanation"])
            st.markdown("### 📊 Model Interpretation")

            if confidence >= 90:
                st.success("The model is highly confident in its prediction.")
            elif confidence >= 75:
                st.warning("The model has moderate confidence. Manual verification is recommended.")
            else:
                st.error("The prediction has low confidence. Please interpret the result with caution.")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Text Contribution",
                    f"{result['text_contribution']:.2f}%"
                )

            with col2:

                st.metric(
                    "Image Contribution",
                    f"{result['image_contribution']:.2f}%"
                )

            st.divider()

            st.subheader("Gate Scores")

            text_gate = result["text_gate"] * 100
            image_gate = result["image_gate"] * 100

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "📝 Text Gate",
                    f"{text_gate:.2f}%"
                )

            with col2:
                st.metric(
                    "🖼 Image Gate",
                    f"{image_gate:.2f}%"
                )

            st.progress(text_gate / 100)
            st.caption(f"Text Gate: {text_gate:.2f}%")

            st.progress(image_gate / 100)
            st.caption(f"Image Gate: {image_gate:.2f}%")

            st.success("Analysis Complete ✅")