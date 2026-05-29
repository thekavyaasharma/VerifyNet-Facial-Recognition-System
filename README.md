# 🔐 VerifyNet — Biometric Identity Verification System

### *A high-security biometric authentication engine powered by Deep Learning*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Kivy](https://img.shields.io/badge/Kivy-Framework-00B0FF?style=for-the-badge&logo=kivy&logoColor=white)](https://kivy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)](https://github.com/thekavyaasharma/Face-Recognition-System)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Automating real-time identity verification with enterprise-level precision using Siamese Neural Networks and one-shot learning.**

<br/>

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Design — Siamese Neural Network](#-model-design--siamese-neural-network)
- [Project Pipeline](#-project-pipeline)
- [Real-Time Verification Flow](#-real-time-verification-flow)
- [Kivy Desktop Application](#-kivy-desktop-application)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Performance Metrics](#-performance-metrics)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 🧠 About the Project

**VerifyNet** is a biometric authentication system that uses **deep learning** to verify human identity in real time. Unlike traditional face recognition systems that require large datasets to identify every new person, VerifyNet employs a **Siamese Neural Network** capable of making accurate identity decisions from a **single reference image** — a paradigm known as **one-shot learning**.

The system captures live webcam input, encodes facial features into high-dimensional embeddings, and computes similarity distances against a verified identity profile. This architecture mirrors the technology used in modern smartphones (like Apple Face ID) and enterprise-grade security systems.

This project was built end-to-end: from raw data collection and preprocessing, through model training and evaluation, to real-time deployment via a cross-platform **Kivy** desktop application.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **One-Shot Learning** | Verifies identity from a single enrolled reference image |
| 📷 **Live Webcam Integration** | Real-time face capture using OpenCV |
| 🧬 **Deep Embedding Engine** | Converts facial images into rich numerical embeddings |
| 📐 **L1 Distance Similarity** | Custom distance layer for precise face comparison |
| ⚡ **Threshold-Based Decision** | Configurable detection and verification thresholds |
| 📊 **F1 + Recall Evaluation** | Robust performance measurement for biometric contexts |
| 💾 **Model Persistence** | Trained model saved and reloaded for deployment |
| 🖥️ **Kivy Desktop App** | Clean cross-platform UI for real-world deployment |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Deep Learning** | TensorFlow / Keras | Model building, training, and saving |
| **Computer Vision** | OpenCV | Webcam capture, image preprocessing |
| **Algorithm** | Siamese Neural Network | One-shot face similarity learning |
| **Data Pipeline** | TF Data API | Efficient batching and preprocessing |
| **UI Framework** | Kivy | Cross-platform desktop application |
| **Notebook** | Jupyter Notebook | Prototyping, training, evaluation |
| **Language** | Python 3.8+ | Core development language |
| **Dataset** | LFW (Labeled Faces in the Wild) | Negative sample source |

---

## 📂 Dataset

The system uses **three categories** of images:

### 1. 🔗 Anchor Images
- Captured in real time via webcam
- Represent the **live input** during verification
- Collected using OpenCV's VideoCapture

### 2. ✅ Positive Images
- Also captured via webcam
- Represent the **enrolled identity** (the same person)
- Stored in `application_data/input_image/`

### 3. ❌ Negative Images
- Sourced from **LFW (Labeled Faces in the Wild)** — an open Kaggle dataset
- Represent **different individuals** to train the model to reject
- Stored in `data/negative/`

> **Data Pairing Strategy:**
> - `(Anchor, Positive)` → Label: `1` (same person)
> - `(Anchor, Negative)` → Label: `0` (different person)

---

## 🧬 Model Design — Siamese Neural Network

The core of VerifyNet is a **Siamese Neural Network**, a specialized deep learning architecture designed for similarity learning.

### Why Siamese Networks?

Traditional classification networks require hundreds of examples per class. Siamese networks learn a *similarity function* — they don't memorize faces, they learn **how to compare them**. This enables:

- ✅ Verification with as **few as one enrollment image**
- ✅ Easy addition of new identities **without retraining**
- ✅ Robustness against unseen faces

### Architecture

```
Input Image (105×105×3)
        │
   ┌────▼────┐
   │ Conv2D  │  64 filters, 10×10, ReLU
   └────┬────┘
        │
   ┌────▼────┐
   │ MaxPool │  64 filters, 2×2
   └────┬────┘
        │
   ┌────▼────┐
   │ Conv2D  │  128 filters, 7×7, ReLU
   └────┬────┘
        │
   ┌────▼────┐
   │ MaxPool │  128 filters, 2×2
   └────┬────┘
        │
   ┌────▼────┐
   │ Conv2D  │  128 filters, 4×4, ReLU
   └────┬────┘
        │
   ┌────▼────┐
   │ MaxPool │  128 filters, 2×2
   └────┬────┘
        │
   ┌────▼────┐
   │ Conv2D  │  256 filters, 4×4, ReLU
   └────┬────┘
        │
   ┌────▼────┐
   │ Flatten │
   └────┬────┘
        │
   ┌────▼────┐
   │ Dense   │  4096 units, Sigmoid
   └─────────┘
   (Embedding Vector)
```

**Both branches share identical weights** — this is what makes it "Siamese" (twin networks).

### L1 Distance Layer (Custom Keras Layer)

```python
class L1Dist(Layer):
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
```

The absolute element-wise difference is passed to a final **Dense(1, sigmoid)** layer that outputs a similarity score between 0 and 1.

### Loss & Optimizer

| Component | Choice | Reason |
|---|---|---|
| **Loss Function** | Binary Cross-Entropy | Binary same/different labels |
| **Optimizer** | Adam | Adaptive learning rate, fast convergence |
| **Training Loop** | Custom `@tf.function` | Handles complex dual-input batches |

---

## 🔄 Project Pipeline

```
Step 1: Data Collection
    └── Capture anchor & positive images via OpenCV webcam
    └── Download LFW dataset for negative samples

Step 2: Preprocessing
    └── Resize all images to 105×105 pixels
    └── Normalize pixel values to [0, 1]
    └── Build (anchor, positive/negative, label) triplets

Step 3: TF Data Pipeline
    └── Map preprocessing function across dataset
    └── Shuffle → Batch → Prefetch for performance

Step 4: Model Architecture
    └── Build embedding subnetwork (CNN)
    └── Build L1 distance layer (custom Keras Layer)
    └── Assemble Siamese model with dual inputs

Step 5: Training
    └── Define binary cross-entropy loss
    └── Write custom training step with GradientTape
    └── Loop over epochs and batches
    └── Monitor loss convergence

Step 6: Evaluation
    └── Generate predictions on test set
    └── Apply 0.5 threshold → binary predictions
    └── Calculate Precision, Recall, F1 Score

Step 7: Save Model
    └── model.save('siamesemodel.h5')

Step 8: Real-Time Deployment
    └── Load saved model
    └── Capture webcam frame
    └── Run 50 verification cycles against positive image pool
    └── Apply detection (0.5) + verification (0.6) thresholds
    └── Output: VERIFIED ✅ or NOT VERIFIED ❌

Step 9: Kivy Application
    └── Build GUI with Kivy framework
    └── Integrate verification function
    └── Deploy as standalone desktop app
```

---

## 📡 Real-Time Verification Flow

The verification function simulates the logic behind **Apple Face ID**:

```python
def verify(model, detection_threshold=0.5, verification_threshold=0.6):
    """
    Runs 50 verification comparisons between webcam input
    and enrolled positive images. Returns VERIFIED if enough
    comparisons pass both thresholds.
    """
    results = []
    for image in os.listdir('application_data/verification_images'):
        input_img  = preprocess('application_data/input_image/input_image.jpg')
        val_img    = preprocess(os.path.join('application_data/verification_images', image))
        result     = model.predict([input_img, val_img])
        results.append(result)

    # Detection: how many comparisons exceeded 0.5 confidence?
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification: did enough comparisons pass? (e.g., 30 out of 50)
    verification = detection / len(os.listdir('application_data/verification_images'))
    verified = verification > verification_threshold

    return results, verified
```

| Threshold | Value | Meaning |
|---|---|---|
| **Detection Threshold** | 0.5 | Minimum model confidence to count a match |
| **Verification Threshold** | 0.6 | Minimum ratio of passed comparisons to confirm identity |

---

## 🖥️ Kivy Desktop Application

The final deployment is wrapped in a **Kivy** cross-platform GUI application.

- **Framework:** Kivy (Python-based, runs on Windows, macOS, Linux, Android)
- **Integration:** Loads the saved TensorFlow model at startup
- **UI Flow:**
  1. User opens the app
  2. Live webcam feed is displayed
  3. User presses **Verify** button
  4. System runs 50-image comparison cycle
  5. Result is shown: ✅ **VERIFIED** or ❌ **UNVERIFIED**

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.8+** installed.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/thekavyaasharma/Face-Recognition-System.git
cd Face-Recognition-System

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow
pip install tensorflow-gpu     # Optional: for GPU acceleration
pip install opencv-python
pip install kivy
pip install numpy matplotlib scikit-learn
```

### Running the Notebook

```bash
jupyter notebook FCR.ipynb
```

Run each cell sequentially:
1. **Section 1–2:** Data collection (webcam will activate)
2. **Section 3–4:** Preprocessing and model building
3. **Section 5:** Training
4. **Section 6:** Evaluation
5. **Section 7:** Real-time verification test
6. **Section 8:** Launch Kivy app

### Running the Application

```bash
cd "Facial Recognition app"
python app.py
```

---

## Project Structure

```
Face-Recognition-System/
│
├──FCR.ipynb                         # Main Jupyter Notebook (full pipeline)
│
├──Facial Recognition app/           # Kivy-based desktop application
│   └── app.py                           # Application entry point
│
├──data/
│   └── negative/                        # LFW negative image samples
│
├──application_data/                 # Runtime data (created during execution)
│   ├── input_image/                     # Live webcam capture stored here
│   └── verification_images/            # 50 enrolled positive images
│
├──requirements.txt                  # Python dependencies
├──project description               # Detailed project notes
├──siamese neural networks.pdf       # Reference research paper
├──.gitignore
└──README.md
```

---

## 📊 Performance Metrics

Model performance is evaluated using metrics appropriate for biometric authentication:

| Metric | Description | Target |
|---|---|---|
| **Precision** | Of all "verified" predictions, how many were correct? | High (minimize false accepts) |
| **Recall** | Of all true positives, how many were caught? | High (minimize false rejects) |
| **F1 Score** | Harmonic mean of precision and recall | Maximized |

> In security-critical systems, **False Accept Rate (FAR)** must be minimized — the model should never verify an impostor. The threshold configuration allows tuning this trade-off based on the deployment context.

---

## Roadmap

- [x] Data collection pipeline (webcam + LFW dataset)
- [x] Image preprocessing and TF data pipeline
- [x] Siamese Neural Network architecture
- [x] Custom L1 Distance layer
- [x] Model training with custom training loop
- [x] Model evaluation (F1, Recall)
- [x] Real-time verification function
- [x] Kivy desktop application integration
- [ ] Multi-face support in a single frame
- [ ] Web API endpoint for remote authentication
- [ ] Mobile application (Android/iOS via Kivy)

---

## References

- **Siamese Neural Networks for One-shot Image Recognition** — Koch, Zemel, Salakhutdinov (2015) *(included as `siamese neural networks.pdf`)*
- **Labeled Faces in the Wild (LFW)** — Kaggle Dataset
- **TensorFlow Documentation** — https://www.tensorflow.org/
- **Kivy Framework** — https://kivy.org/

---

## 👩‍💻 Author

<div align="center">

**Kavya Sharma**

[![GitHub](https://img.shields.io/badge/GitHub-thekavyaasharma-181717?style=for-the-badge&logo=github)](https://github.com/thekavyaasharma)

*Built with passion for AI, Deep Learning, and real-world impact.*

</div>

---

<div align="center">

*⭐ If you found this project useful or interesting, consider giving it a star!*

</div>
