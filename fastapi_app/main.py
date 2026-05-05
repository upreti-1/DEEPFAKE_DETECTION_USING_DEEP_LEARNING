import os
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_60 = ROOT_DIR / "model_ff_data/model_97_acc_60_frames_FF_data.pt"
DEFAULT_MODEL_100 = ROOT_DIR / "model_ff_data/model_97_acc_100_frames_FF_data.pt"
DEFAULT_MODEL_PATH = (
    DEFAULT_MODEL_100
    if DEFAULT_MODEL_100.exists()
    else DEFAULT_MODEL_60
    if DEFAULT_MODEL_60.exists()
    else DEFAULT_MODEL_100
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found at {MODEL_PATH}. Set MODEL_PATH environment variable or place the model file at the repository root."
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SM = nn.Softmax(dim=1)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "gif", "webm", "avi", "3gp", "wmv", "flv", "mkv"}

app = FastAPI(title="Deepfake Detection FastAPI")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


def get_sequence_length_from_model_path(path: Path) -> int:
    match = re.search(r"_(\d+)_frames_", path.name)
    if match:
        return int(match.group(1))
    return 60


class DeepfakeModel(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = 2048, lstm_layers: int = 1, hidden_dim: int = 2048, bidirectional: bool = False):
        super().__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim if bidirectional else latent_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


class VideoSequenceDataset(Dataset):
    def __init__(self, video_paths, sequence_length=60, transform=None):
        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = []
        for frame in self.frame_extract(video_path):
            frames.append(self.process_frame(frame))
            if len(frames) == self.sequence_length:
                break

        if len(frames) == 0:
            raise ValueError("No valid frames could be extracted from the uploaded video.")

        while len(frames) < self.sequence_length:
            frames.append(frames[-1])

        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        video_capture = cv2.VideoCapture(str(path))
        success = True
        while success:
            success, image = video_capture.read()
            if success and image is not None:
                yield image
        video_capture.release()

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(rgb_frame)


def load_model(path: Path) -> nn.Module:
    model = DeepfakeModel(2)
    state = torch.load(str(path), map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model


def compute_prediction(model: nn.Module, tensor: torch.Tensor):
    with torch.no_grad():
        _, logits = model(tensor.to(DEVICE))
        probabilities = SM(logits)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = float(probabilities[0, prediction].item() * 100.0)
    return prediction, confidence


@app.on_event("startup")
async def startup_event():
    print(f"Loading model from: {MODEL_PATH}")
    app.state.model = load_model(MODEL_PATH)
    app.state.default_sequence_length = get_sequence_length_from_model_path(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_name": MODEL_PATH.name,
            "device": DEVICE.type,
            "default_sequence_length": app.state.default_sequence_length,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...), sequence_length: int = Form(None)):
    if file.filename == "":
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_name": MODEL_PATH.name,
                "device": DEVICE.type,
                "error": "Please upload a video file.",
            },
        )

    extension = file.filename.rsplit(".", 1)[-1].lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_name": MODEL_PATH.name,
                "device": DEVICE.type,
                "error": "Unsupported file format. Supported formats: mp4, avi, mkv, webm, gif, 3gp, wmv, flv.",
            },
        )

    if sequence_length is None:
        sequence_length = app.state.default_sequence_length

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_path = Path(tmp_dir) / file.filename
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            dataset = VideoSequenceDataset([saved_path], sequence_length=sequence_length, transform=train_transforms)
            video_tensor = dataset[0]
            prediction, confidence = compute_prediction(app.state.model, video_tensor)
            label = "REAL" if prediction == 1 else "FAKE"
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "model_name": MODEL_PATH.name,
                    "device": DEVICE.type,
                    "output": label,
                    "confidence": f"{confidence:.1f}",
                    "sequence_length": sequence_length,
                },
            )
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "model_name": MODEL_PATH.name,
                    "device": DEVICE.type,
                    "error": f"Prediction failed: {exc}",
                },
            )
