import os
import cv2
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import deque
from dotenv import load_dotenv
from transformers import CLIPModel, CLIPProcessor

# Load settings and secrets from a local .env file (never committed).
load_dotenv()


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "isl_temporal_transformer_best.pt")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.json")
CLIP_NAME = os.getenv("CLIP_NAME", "openai/clip-vit-base-patch32")

# Optional: Hugging Face token (only needed for private models / higher rate limits).
HF_TOKEN = os.getenv("HF_TOKEN") or None

MAX_FRAMES = int(os.getenv("MAX_FRAMES", "300"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "10"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
SAMPLE_INTERVAL = float(os.getenv("SAMPLE_INTERVAL", "0.1"))        # ~10 frames per second
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL", "2.0"))  # Predict every 2 seconds
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# ============================================================
# 2. LOAD LABELS
# ============================================================

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_to_id = json.load(f)

if isinstance(label_to_id, list):
    id_to_label = {
        i: label for i, label in enumerate(label_to_id)
    }
else:
    id_to_label = {
        int(class_id): label
        for label, class_id in label_to_id.items()
    }

NUM_CLASSES = len(id_to_label)

if NUM_CLASSES == 0:
    raise ValueError("No class labels found in labels.json")

print("Number of classes:", NUM_CLASSES)


# ============================================================
# 3. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# Handle checkpoints saved from DataParallel.
if any(k.startswith("module.") for k in state_dict):
    state_dict = {
        k.removeprefix("module."): v
        for k, v in state_dict.items()
    }

# Infer architecture dimensions from saved tensors where possible.
required_keys = [
    "projection.weight",
    "position_embedding",
    "temporal_encoder.layers.0.self_attn.in_proj_weight",
    "temporal_encoder.layers.0.linear1.weight",
    "classifier.0.weight",
    "classifier.3.weight",
]

missing = [k for k in required_keys if k not in state_dict]
if missing:
    raise RuntimeError(
        "Checkpoint format differs from the expected training model.\n"
        f"Missing checkpoint keys: {missing}\n"
        "Use the exact model class and checkpoint from training."
    )

input_dim = state_dict["projection.weight"].shape[1]
d_model = state_dict["projection.weight"].shape[0]
max_frames = state_dict["position_embedding"].shape[1]
ff_dim = state_dict["temporal_encoder.layers.0.linear1.weight"].shape[0]
classifier_hidden = state_dict["classifier.0.weight"].shape[0]
checkpoint_classes = state_dict["classifier.3.weight"].shape[0]

num_layers = sum(
    1 for k in state_dict
    if k.startswith("temporal_encoder.layers.")
    and k.endswith(".self_attn.in_proj_weight")
)

if checkpoint_classes != NUM_CLASSES:
    raise ValueError(
        f"Checkpoint has {checkpoint_classes} output classes, "
        f"but labels.json has {NUM_CLASSES} labels. "
        "Use the labels.json saved with this exact checkpoint."
    )

if max_frames != MAX_FRAMES:
    MAX_FRAMES = max_frames
    print("Using checkpoint frame length:", MAX_FRAMES)

print(
    f"Architecture: input={input_dim}, d_model={d_model}, "
    f"layers={num_layers}, FFN={ff_dim}, classes={checkpoint_classes}"
)


# ============================================================
# 4. TEMPORAL TRANSFORMER
#    Layer names must match the saved checkpoint.
# ============================================================

class TemporalTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        num_heads,
        num_layers,
        ff_dim,
        num_classes,
        max_frames,
        classifier_hidden,
    ):
        super().__init__()

        self.projection = nn.Linear(input_dim, d_model)

        self.position_embedding = nn.Parameter(
            torch.randn(1, max_frames, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, features, attention_mask):
        x = self.projection(features)

        x = x + self.position_embedding[:, :x.size(1), :]

        padding_mask = attention_mask == 0

        x = self.temporal_encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        mask = attention_mask.unsqueeze(-1).float()

        x = (x * mask).sum(dim=1)
        x = x / mask.sum(dim=1).clamp(min=1)

        x = self.norm(x)

        return self.classifier(x)


# Number of attention heads must match training.
NUM_HEADS = 8

if d_model % NUM_HEADS != 0:
    raise ValueError(
        f"d_model={d_model} is incompatible with {NUM_HEADS} heads. "
        "Set NUM_HEADS to the value used during training."
    )

model = TemporalTransformerClassifier(
    input_dim=input_dim,
    d_model=d_model,
    num_heads=NUM_HEADS,
    num_layers=num_layers,
    ff_dim=ff_dim,
    num_classes=checkpoint_classes,
    max_frames=MAX_FRAMES,
    classifier_hidden=classifier_hidden,
).to(DEVICE)

model.load_state_dict(state_dict, strict=True)
model.eval()

print("Temporal Transformer checkpoint loaded successfully.")


# ============================================================
# 5. LOAD THE SAME CLIP ENCODER USED DURING TRAINING
# ============================================================

clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME, token=HF_TOKEN)
clip_model = CLIPModel.from_pretrained(CLIP_NAME, token=HF_TOKEN).to(DEVICE)
clip_model.eval()

for parameter in clip_model.parameters():
    parameter.requires_grad = False

if clip_model.config.projection_dim != input_dim:
    raise ValueError(
        f"CLIP outputs {clip_model.config.projection_dim} dimensions, "
        f"but checkpoint expects {input_dim}."
    )

print("CLIP encoder loaded successfully.")


# ============================================================
# 6. FRAME PREPROCESSING AND CLIP FEATURES
# ============================================================

def preprocess_frame(frame):
    # Webcam gives BGR; CLIP processor expects RGB images.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    return frame


@torch.inference_mode()
def extract_clip_features(frames):
    all_embeddings = []

    for start in range(0, len(frames), CHUNK_SIZE):
        chunk = frames[start:start + CHUNK_SIZE]

        inputs = clip_processor(
            images=chunk,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(DEVICE)

        vision_output = clip_model.vision_model(
            pixel_values=pixel_values
        )

        pooled_features = vision_output.pooler_output

        embeddings = clip_model.visual_projection(pooled_features)

        # Same L2 normalization used during feature extraction.
        embeddings = F.normalize(
            embeddings.float(),
            p=2,
            dim=-1,
        )

        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)


# ============================================================
# 7. LIVE WEBCAM INFERENCE
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Close other camera applications "
        "or try CAMERA_INDEX = 1."
    )

frame_buffer = deque(maxlen=MAX_FRAMES)

last_sample_time = 0.0
last_prediction_time = 0.0

last_prediction = "Waiting for sign..."
last_confidence = 0.0
top_predictions = []

print("\nWebcam is running.")
print("Show one sign clearly and wait for prediction.")
print("Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read a webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        # Sample approximately 10 frames per second.
        if now - last_sample_time >= SAMPLE_INTERVAL:
            frame_buffer.append(preprocess_frame(frame))
            last_sample_time = now

        # Do not infer until enough frames have been collected.
        if (
            len(frame_buffer) >= 10
            and now - last_prediction_time >= INFERENCE_INTERVAL
        ):
            try:
                frames = list(frame_buffer)

                # Keep at most MAX_FRAMES, evenly spread if necessary.
                if len(frames) > MAX_FRAMES:
                    indices = np.linspace(
                        0, len(frames) - 1, MAX_FRAMES
                    ).astype(int)
                    frames = [frames[i] for i in indices]

                features = extract_clip_features(frames)

                valid_count = features.shape[0]

                padded_features = torch.zeros(
                    (1, MAX_FRAMES, input_dim),
                    dtype=torch.float32,
                    device=DEVICE,
                )

                attention_mask = torch.zeros(
                    (1, MAX_FRAMES),
                    dtype=torch.long,
                    device=DEVICE,
                )

                padded_features[0, :valid_count] = features
                attention_mask[0, :valid_count] = 1

                logits = model(padded_features, attention_mask)
                probabilities = torch.softmax(logits, dim=1)[0]

                top_k = min(3, NUM_CLASSES)
                values, indices = torch.topk(probabilities, k=top_k)

                top_predictions = [
                    (
                        id_to_label.get(int(idx), "Unknown"),
                        float(score),
                    )
                    for score, idx in zip(
                        values.cpu().tolist(),
                        indices.cpu().tolist(),
                    )
                ]

                last_prediction = top_predictions[0][0]
                last_confidence = top_predictions[0][1]
                last_prediction_time = now

            except Exception as exc:
                print("Inference error:", repr(exc))
                last_prediction_time = now

        # Draw result overlay.
        cv2.rectangle(
            frame,
            (10, 10),
            (frame.shape[1] - 10, 155),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"Prediction: {last_prediction}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {last_confidence * 100:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        for i, (label, score) in enumerate(top_predictions[:3]):
            cv2.putText(
                frame,
                f"{i + 1}. {label}: {score * 100:.1f}%",
                (20, 110 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("ISL Live Sign Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")
