# 🕵️ Deepfake Detection Using Deep Learning

## 📌 Overview
Deepfakes are AI-generated synthetic media that can convincingly mimic real people, posing risks in misinformation, fraud, and digital security. This project implements **deep learning models** to detect deepfake videos and images, providing a framework for researchers and developers to combat malicious use of this technology.

---

## 🚀 Features
- End-to-end deepfake detection pipeline
- Video frame extraction and preprocessing
- CNN and transfer learning architectures
- Configurable training scripts
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Modular codebase for experimentation

---

## 🛠️ Tech Stack
- **Language:** Python 3.x  
- **Frameworks:** TensorFlow / Keras, PyTorch  
- **Libraries:** NumPy, OpenCV, Matplotlib, Scikit-learn  
- **Datasets:** FaceForensics++, Celeb-DF, custom datasets  

---

## 📂 Project Structure
DEEPFAKE_DETECTION_USING_DEEP_LEARNING/
│── data/                # Dataset storage
│── notebooks/           # Jupyter notebooks for experiments
│── src/                 # Core source code
│   ├── preprocessing/   # Data cleaning & frame extraction
│   ├── models/          # Deep learning architectures
│   ├── training/        # Training scripts
│   └── evaluation/      # Metrics & performance evaluation
│── results/             # Saved models & evaluation outputs
│── requirements.txt     # Dependencies
│── README.md            # Documentation



---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/upreti-1/DEEPFAKE_DETECTION_USING_DEEP_LEARNING.git
cd DEEPFAKE_DETECTION_USING_DEEP_LEARNING
pip install -r requirements.txt


📊 Usage
1. Preprocess Dataset
python src/preprocessing/preprocess.py --input data/raw --output data/processed

2. Train Model
python src/training/train.py --model cnn --epochs 50 --batch_size 32

3. Evaluate Model
python src/evaluation/evaluate.py --model saved_model.h5 --test data/processed/test

📈 Results
Robust detection of manipulated frames
Training curves and confusion matrices available in results/

🙌 Acknowledgements
FaceForensics++ Dataset
Celeb-DF Dataset
Open-source deep learning community
