# Bi-Directional Indian Sign Language (ISL) – Speech Translation System

### AI-Powered Sign Language Recognition Using CLIP and Temporal Transformers

<p align="center">
  <b>Bridging the communication gap between Deaf and hearing communities through Deep Learning.</b>
</p>

---

## 🚀 Project Overview

Communication between Indian Sign Language (ISL) users and non-signers remains a significant accessibility challenge, particularly in healthcare, education, and public services.

This project focuses on developing a **Bi-Directional Multimodal Sign Language–Speech Translation System** that leverages Deep Learning and Transformer architectures to recognize Indian Sign Language gestures from video and facilitate communication through English text and speech.

The implemented recognition pipeline combines a pretrained **CLIP Image Encoder** with a **Temporal Transformer Encoder** to extract visual representations, capture temporal dependencies, and classify sign-language videos.

The broader proposed architecture aims to extend this functionality towards sentence-level translation, speech synthesis, emotion awareness, context awareness, and reverse speech-to-sign communication.

## 🎯 Key Objectives

- Recognize Indian Sign Language gestures from video input.
- Extract meaningful visual representations using pretrained CLIP.
- Capture temporal relationships between consecutive video frames.
- Classify sign-language videos into corresponding sign categories.
- Develop a foundation for English sentence generation and speech output.
- Explore context-aware and emotion-aware multimodal communication.

## 🧠 System Architecture

The proposed architecture follows a modular deep-learning pipeline:

```text
          ISL Video Dataset
                  |
                  v
          Video Processing
                  |
                  v
       Uniform Frame Sampling
             (300 Frames)
                  |
                  v
          Frame Preprocessing
                  |
                  v
          CLIP Image Encoder
          (Visual Features)
                  |
                  v
          L2 Normalization
                  |
                  v
          Linear Projection
                  |
                  v
       Temporal Positional Encoding
                  |
                  v
       Temporal Transformer Encoder
          (Temporal Learning)
                  |
             +----+----+
             |         |
             v         v
       Stage 1      Stage 2
     Classification  Transformer
         Head         Decoder
             |         |
             v         v
       Predicted    English Text
       Sign Class       |
                       v
                 Text-to-Speech
```

**Core architectural components:**

1. **CLIP Image Encoder:** Extracts semantic visual features from individual video frames using pretrained image representations.
2. **L2 Normalization:** Normalizes frame embeddings to maintain consistent feature magnitudes.
3. **Linear Projection:** Maps CLIP embeddings into the configured Transformer model dimension.
4. **Temporal Positional Encoding:** Introduces frame-order information into the sequence.
5. **Temporal Transformer Encoder:** Uses multi-head self-attention to learn relationships between frames.
6. **Classification Head:** Predicts the corresponding ISL sign category.
7. **Transformer Decoder (Proposed):** Designed to generate meaningful English sentences using encoded visual features.

## ⚙️ Technical Implementation

### 1. Video Processing

Each input video is converted into a fixed-length temporal representation.

- Uniformly samples exactly 300 frames.
- Pads shorter videos by repeating frames.
- Applies RGB conversion, resizing, and CLIP-compatible preprocessing.
- Processes frames in chunks of 10 to manage GPU memory.

### 2. Visual Feature Extraction

A pretrained CLIP image encoder extracts semantic embeddings from each frame.

The resulting embeddings are L2-normalized before being passed to the temporal modeling network.

### 3. Temporal Learning

The Temporal Transformer Encoder captures dependencies across the complete video sequence through:

- Multi-Head Self-Attention
- Scaled Dot-Product Attention
- Positional Encoding
- Residual Connections
- Layer Normalization
- Feed-Forward Networks

### 4. Sign Classification

The encoded temporal representations are passed through a classification head to predict the corresponding sign class.

### 5. Proposed Sentence Generation

A Transformer Decoder is designed to generate English sentences through masked self-attention and cross-attention over encoded video representations.

This stage requires paired sentence-level ISL video and English translation data.

## 📊 Model Evaluation Results

The implemented Stage 1 classification model achieved the following evaluation metrics:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | **85.42%** |
| Precision | **80.73%** |
| Recall | **85.42%** |
| F1-Score | **82.08%** |

These results demonstrate the performance of the current sign-classification stage.

*Note: The reported metrics correspond to Stage 1 classification. Sentence-level translation metrics such as BLEU, ROUGE, and WER have not yet been reported.*

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Visual Representation | CLIP |
| Architecture | Temporal Transformer |
| Attention Mechanism | Multi-Head Self-Attention |
| Video Processing | OpenCV, FFmpeg-compatible tools |
| Data Processing | NumPy, Pandas |
| GPU Training | CUDA, College HPC |
| Job Scheduling | PBS |
| Dataset Access | Google Drive, rclone |
| Version Control | Git, GitHub |

## 📂 Dataset

The project utilizes an Indian Sign Language dictionary video dataset organized into multiple collections.

The dataset includes:

- Alphabet signs (A–Z)
- Number signs
- New 2500 ISL Dictionary Videos
- NCERT-related sign collections
- MHSL collection
- Other available ISL dictionary collections

**Dataset characteristics:**
- Primarily word/sign-level video samples.
- Multiple collections with different organizational structures.
- Metadata generation and label validation before training.
- Fixed-length frame sampling for consistent model inputs.

*Dataset access and redistribution depend on the original dataset's licensing and permissions. The dataset is not included in this repository.*

## 📈 Development Roadmap

| Feature | Status |
|---|---|
| ISL Video Dataset Processing | Implemented |
| Metadata Generation and Label Validation | Implemented |
| 300-Frame Sampling | Implemented |
| CLIP Feature Extraction | Implemented |
| L2 Embedding Normalization | Implemented |
| Temporal Transformer Encoder | Implemented |
| Stage 1 Sign Classification | Implemented |
| Model Evaluation | Initial results available |
| Sentence-Level Translation | Proposed |
| Text-to-Speech Integration | Proposed |
| Context Awareness | Future |
| Emotion Awareness | Future |
| Speech-to-Sign Translation | Future |
| Real-Time Translation | Future |

## 🔬 Research and Technical Contributions

The project explores an integrated architecture combining pretrained visual representations with explicit temporal modeling for Indian Sign Language recognition.

Key engineering aspects include:

- Integration of pretrained CLIP visual representations with a dedicated Temporal Transformer.
- Fixed-length temporal representation using 300 sampled frames.
- Memory-efficient chunk-based visual feature extraction.
- L2-normalized visual embeddings.
- Modular two-stage training strategy.
- Evaluation using accuracy, precision, recall, and F1-score.

The contribution is the proposed system integration and implementation, rather than claiming the underlying CLIP or Transformer architectures as novel.

## 🌍 Potential Applications

- Inclusive education and digital learning.
- Healthcare communication assistance.
- Accessibility at government service centers.
- Communication between ISL users and non-signers.
- Assistive technologies and human-computer interaction.

## 🔮 Future Enhancements

1. Train the Transformer Decoder using paired sentence-level ISL datasets.
2. Integrate a Text-to-Speech engine for spoken English output.
3. Introduce facial expression and body-language features for emotion awareness.
4. Incorporate conversational context for better interpretation.
5. Develop reverse speech-to-sign translation.
6. Explore real-time inference and deployment.

## 👨‍💻 Project Information

**Project:** Bi-Directional Multimodal Sign Language–Speech Translation System with Emotion & Context Awareness

**Domain:** Artificial Intelligence | Deep Learning | Computer Vision | Natural Language Processing

**Specialization:** B.Tech – Computer Science and Engineering (Data Science)

**Institution:** UPES, Dehradun

**Project Type:** Final-Year Major Project

## 🤝 Contributions

Contributions, suggestions, and research discussions are welcome.

If you are interested in improving Indian Sign Language recognition and accessible AI technologies, feel free to explore the project and contribute.

## ⭐ Support

If you find this project interesting, consider giving the repository a star!

---

**Building AI for a more accessible and inclusive world.**
