# Bi-Directional Indian Sign Language (ISL) – Speech Translation System

**AI-Powered Sign Language Recognition Using CLIP and Temporal Transformers**

Bridging the communication gap between Deaf and hearing communities through Deep Learning.

---

## 1. Project Overview

Communication between Indian Sign Language (ISL) users and non-signers remains a significant accessibility challenge, particularly in healthcare, education, and public services.

This project focuses on developing a **Bi-Directional Multimodal Sign Language–Speech Translation System** that leverages Deep Learning and Transformer architectures to recognize Indian Sign Language gestures from video and facilitate communication through English text and speech.

The implemented recognition pipeline integrates a pretrained **CLIP Image Encoder** with a **Temporal Transformer Encoder** to extract visual representations, capture temporal dependencies, and classify sign-language videos.

The broader proposed architecture aims to extend this functionality through sentence-level translation, speech synthesis, emotion awareness, context awareness, and reverse speech-to-sign communication.

## 2. Project Objectives

- Recognize Indian Sign Language gestures from video input.
- Extract meaningful visual features using pretrained CLIP.
- Capture temporal relationships between consecutive video frames.
- Classify sign-language videos into corresponding sign categories.
- Develop a foundation for English sentence generation and speech output.
- Explore context-aware and emotion-aware multimodal communication.
- Work towards bidirectional communication between ISL users and non-signers.

## 3. System Architecture

The system follows a modular deep-learning pipeline combining visual feature extraction, temporal modeling, and language generation.

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
          Stage 1       Stage 2
       Classification  Transformer
            Head        Decoder
                |         |
                v         v
        Predicted Sign  English Text
                           |
                           v
                    Text-to-Speech
```

### Core Components

**1. CLIP Image Encoder**

Uses pretrained visual representations to extract semantic embeddings from individual video frames.

**2. L2 Normalization**

Normalizes frame embeddings to maintain consistent feature magnitudes and improve numerical stability.

**3. Linear Projection**

Maps CLIP embeddings into the configured Transformer model dimension.

**4. Temporal Positional Encoding**

Introduces sequential information so that the model can understand the order of video frames.

**5. Temporal Transformer Encoder**

Uses Multi-Head Self-Attention to capture relationships and dependencies across the complete video sequence.

**6. Classification Head**

Processes encoded temporal representations to predict the corresponding ISL sign category.

**7. Transformer Decoder (Proposed)**

Designed to generate meaningful English sentences using masked self-attention and cross-attention over encoded video representations.

## 4. Technical Implementation

### Video Processing

Each input video is converted into a fixed-length temporal representation.

- Uniformly samples exactly 300 frames.
- Pads shorter videos by repeating frames.
- Converts frames into RGB format.
- Applies resizing and CLIP-compatible normalization.
- Processes frames in chunks of 10 to manage GPU memory.

### Visual Feature Extraction

A pretrained CLIP Image Encoder extracts semantic embeddings from each frame.

The extracted embeddings undergo L2 normalization before being passed to the temporal modeling network.

### Temporal Learning

The Temporal Transformer Encoder captures dependencies across video frames through:

- Multi-Head Self-Attention
- Scaled Dot-Product Attention
- Temporal Positional Encoding
- Residual Connections
- Layer Normalization
- Feed-Forward Networks

### Sign Classification

The encoded temporal representations are passed through a classification head to predict the corresponding sign category.

### Proposed Sentence Generation

A Transformer Decoder is designed to generate English sentences through autoregressive token generation, masked self-attention, and cross-attention over visual-temporal representations.

This stage requires paired sentence-level ISL videos and English translations.

## 5. Model Evaluation Results

The implemented Stage 1 classification model achieved the following evaluation results:

| Evaluation Metric | Performance |
|---|---:|
| Accuracy | **85.42%** |
| Precision | **80.73%** |
| Recall | **85.42%** |
| F1-Score | **82.08%** |

These results represent the initial performance of the implemented sign classification model.

**Note:** The reported metrics correspond to Stage 1 sign classification. Sentence-level translation metrics such as BLEU, ROUGE, and Word Error Rate (WER) have not yet been reported.

## 6. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Visual Feature Extraction | CLIP |
| Model Architecture | Temporal Transformer |
| Attention Mechanism | Multi-Head Self-Attention |
| Video Processing | OpenCV, FFmpeg-compatible tools |
| Data Processing | NumPy, Pandas |
| GPU Computing | CUDA |
| Training Infrastructure | College HPC, PBS |
| Dataset Access | Google Drive, rclone |
| Version Control | Git, GitHub |

## 7. Dataset

The project utilizes an Indian Sign Language dictionary video dataset organized into multiple collections.

The dataset includes:

- Alphabet signs (A–Z)
- Number signs
- New 2500 ISL Dictionary Videos
- NCERT-related sign collections
- MHSL collection
- Other available ISL dictionary collections

### Dataset Characteristics

- Primarily word/sign-level video samples.
- Multiple collections with different organizational structures.
- Metadata generation and label validation.
- Fixed-length frame sampling for consistent model inputs.
- Video validity and class-distribution analysis.

The dataset is stored externally in Google Drive and is not included in this repository.

Dataset access and redistribution are subject to the original dataset licensing and permissions.

## 8. Implementation Status

| Feature | Status |
|---|---|
| ISL Dataset Processing | Implemented |
| Metadata Generation | Implemented |
| Label Validation | Implemented |
| 300-Frame Sampling | Implemented |
| CLIP Feature Extraction | Implemented |
| L2 Embedding Normalization | Implemented |
| Temporal Transformer Encoder | Implemented |
| Stage 1 Sign Classification | Implemented |
| Model Evaluation | Initial Results Available |
| Sentence-Level Translation | Proposed |
| Text-to-Speech Integration | Proposed |
| Context Awareness | Future |
| Emotion Awareness | Future |
| Speech-to-Sign Translation | Future |
| Real-Time Translation | Future |

## 9. Research and Technical Contributions

The project explores the integration of pretrained visual representations with explicit temporal modeling for Indian Sign Language recognition.

Key engineering contributions include:

- Integration of pretrained CLIP visual representations with a dedicated Temporal Transformer.
- Fixed-length temporal representation using 300 sampled frames.
- Memory-efficient chunk-based visual feature extraction.
- L2-normalized visual embeddings.
- Temporal positional encoding for sequential understanding.
- Modular two-stage training strategy.
- Evaluation using accuracy, precision, recall, and F1-score.

The contribution focuses on the proposed system integration and implementation rather than claiming the underlying CLIP or Transformer architectures as novel.

## 10. Potential Applications

- Inclusive education and digital learning.
- Healthcare communication assistance.
- Accessibility at government service centers.
- Communication between ISL users and non-signers.
- Assistive technologies.
- Human-computer interaction.

## 11. Future Enhancements

1. Train the Transformer Decoder using paired sentence-level ISL datasets.
2. Integrate a Text-to-Speech engine for spoken English output.
3. Introduce facial expression and body-language features for emotion awareness.
4. Incorporate conversational context for better interpretation.
5. Develop reverse speech-to-sign translation.
6. Explore real-time video inference.
7. Improve model generalization across different signing styles and environments.

## 12. Project Information

| Details | Description |
|---|---|
| Project Title | Bi-Directional Multimodal Sign Language–Speech Translation System with Emotion & Context Awareness |
| Domain | Artificial Intelligence and Deep Learning |
| Specialization | B.Tech – Computer Science and Engineering (Data Science) |
| Institution | UPES, Dehradun |
| Project Type | Final-Year Major Project |

## 13. Contributions

Contributions, suggestions, and research discussions are welcome.

If you are interested in improving Indian Sign Language recognition and developing accessible AI technologies, feel free to explore the project and contribute.

## 14. Acknowledgements

We acknowledge the research community behind CLIP, Transformer architectures, and Indian Sign Language resources that provide the foundation for this work.

---

**Building AI for a more accessible and inclusive world.**
