# Deepfake Detection API

This folder contains a standalone FastAPI application for deepfake video detection using a ResNeXt50 + LSTM model.

## Contents

- `main.py` — FastAPI application and model inference logic
- `requirements.txt` — Python dependencies for the app
- `templates/index.html` — web upload interface
- `static/` — optional frontend assets

## Model files

The app expects model weights in the project root under `model_ff_data/`:

- `model_ff_data/model_97_acc_100_frames_FF_data.pt`
- `model_ff_data/model_97_acc_60_frames_FF_data.pt` (optional)

The app selects the 100-frame model by default if available.

## Requirements

Install the required packages inside your Python environment:

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python -m uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- `http://127.0.0.1:8000`

## Usage

- Upload a video on the page
- Optionally set the sequence length (defaults to the model's expected frame count)
- The app returns `REAL` or `FAKE` with a confidence score

## Notes

- If model weights are large, consider using Git LFS for `.pt` files.
- The app is independent of the Django application.
- If you want to force a specific model, set the `MODEL_PATH` environment variable before running.

```bash
set MODEL_PATH=C:\Users\acer\OneDrive\Desktop\Deepfake_detection_using_deep_learning\model_ff_data\model_97_acc_100_frames_FF_data.pt
python -m uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000
```
