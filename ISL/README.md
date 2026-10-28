# ISL Live Sign Classification

Real-time **Indian Sign Language (ISL)** word recognition from a webcam, using frozen
CLIP image features and a small temporal Transformer classifier.

## How it works

1. The webcam is sampled at ~10 frames per second into a rolling buffer (up to 300 frames).
2. Each frame is encoded with **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) into an L2-normalised 512-d embedding.
3. The sequence of embeddings is passed through a **Temporal Transformer** (`isl_temporal_transformer_best.pt`) which predicts one of **48 sign classes**.
4. The top-3 predictions and confidence scores are drawn on the live video feed every 2 seconds.

## Supported signs

The 48 classes are listed in [`labels.json`](labels.json) (e.g. `amazon`, `diwali`, `family`, `india`, `peacock`, `volcano`, `website`, ...).

## Setup

Requires Python 3.10+ and a webcam.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> For GPU acceleration, install the CUDA build of PyTorch first from <https://pytorch.org/get-started/locally/>, then run `pip install -r requirements.txt`.

## Usage

```bash
python live_test.py
```

- Show **one sign clearly** in front of the camera and wait for the next prediction.
- Press **Q** to quit.
- If the camera does not open, close other apps using it or change `CAMERA_INDEX` in `live_test.py`.

The CLIP weights are downloaded automatically from Hugging Face on first run.

## Configuration

Settings and secrets are read from a local `.env` file (git-ignored). Copy the template and edit it:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Never commit `.env`; put API keys, tokens, emails or phone numbers there only. Available variables:

| Setting | Default | Meaning |
| --- | --- | --- |
| `SAMPLE_INTERVAL` | `0.1` s | Time between buffered frames (~10 FPS) |
| `INFERENCE_INTERVAL` | `2.0` s | How often a prediction is made |
| `CAMERA_INDEX` | `0` | Which webcam to use |
| `MAX_FRAMES` | `300` | Max sequence length (read from the checkpoint) |
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token |
| `CHECKPOINT_PATH`, `LABELS_PATH`, `CLIP_NAME` | see `.env.example` | Model/file locations |

## Files

| File | Description |
| --- | --- |
| `live_test.py` | Live webcam inference script |
| `isl_temporal_transformer_best.pt` | Trained Temporal Transformer checkpoint (~10 MB) |
| `labels.json` | Class name → index mapping |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your local `.env` |

## Notes

- `torch.load(..., weights_only=False)` is used to load the checkpoint. Only load checkpoints you trust.
- Accuracy depends on lighting, background and framing; best results come from a plain background with the signer centred.
