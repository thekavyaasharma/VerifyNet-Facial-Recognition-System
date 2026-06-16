# 🔐 VerifyNet — Biometric Identity Verification System

### *Real-time face authentication powered by Siamese Neural Networks and one-shot learning*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Kivy](https://img.shields.io/badge/Kivy-Framework-00B0FF?style=for-the-badge&logo=kivy&logoColor=white)](https://kivy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)](https://github.com/thekavyaasharma/Face-Recognition-System)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Built end-to-end — from raw webcam data collection and model training to real-time deployment — VerifyNet verifies identity from a single reference image using the same one-shot learning paradigm behind Apple Face ID.**

<br/>

---

## 🧠 What This Project Does (and Why It's Hard)

Most face recognition systems need **hundreds of images per person** to work. VerifyNet solves a harder problem: verifying identity from **just one enrolled image**.

This is achieved using a **Siamese Neural Network** — an architecture that doesn't memorize faces, but instead learns *how to compare* them. Two identical CNN branches (sharing weights) process a live capture and a stored reference image, then a custom **L1 Distance Layer** computes the similarity between their embeddings.

The result is a system that:
- Works with **a single reference photo** per identity
- Can add new users **without retraining the model**
- Runs in **real time** via webcam using a cross-platform desktop GUI

This project was built completely from scratch: custom data collection, a hand-written training loop using `GradientTape`, and a deployed Kivy application — no shortcuts.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **One-Shot Learning** | Verifies identity from a single enrolled reference image — no large per-person dataset needed |
| 📷 **Live Webcam Integration** | Real-time face capture and preprocessing using OpenCV |
| 🧬 **Deep Embedding Engine** | Custom 4-layer CNN encodes facial features into 4096-dim vectors |
| 📐 **L1 Distance Similarity** | Custom Keras layer computes element-wise absolute difference between embeddings |
| ⚡ **Dual-Threshold Decision** | Configurable detection (0.5) + verification (0.6) thresholds for security tuning |
| 📊 **Precision / Recall / F1** | Evaluation suite designed for biometric authentication tradeoffs |
| 💾 **Model Persistence** | Trained model saved as `.h5` and reloaded at deployment time |
| 🖥️ **Kivy Desktop App** | Clean cross-platform GUI — runs on Windows, macOS, and Linux |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Deep Learning** | TensorFlow / Keras | Model architecture, training, and saving |
| **Computer Vision** | OpenCV | Webcam capture, image resizing, normalization |
| **Algorithm** | Siamese Neural Network | One-shot similarity learning |
| **Data Pipeline** | TF Data API | Shuffle → Batch → Prefetch for fast iteration |
| **UI Framework** | Kivy | Cross-platform desktop deployment |
| **Experimentation** | Jupyter Notebook | Prototyping, training, evaluation |
| **Language** | Python 3.8+ | End-to-end development |
| **Dataset** | LFW (Labeled Faces in the Wild) | Negative (impostor) sample source |

---

## 📂 Dataset Strategy

The model is trained on **image pairs with binary labels**, not individual identities:

| Pair Type | Images | Label | Meaning |
|---|---|---|---|
| **(Anchor, Positive)** | Same person | `1` | Match — verify |
| **(Anchor, Negative)** | Different person | `0` | No match — reject |

**Sources:**
- **Anchor + Positive images** — Captured live via webcam (real enrollment scenario)
- **Negative images** — LFW dataset (Kaggle), representing the universe of impostors

This pairing strategy teaches the model to recognize *similarity*, not identities — the key advantage of the Siamese approach.

---

## 🧬 Model Architecture

### Embedding Subnetwork (shared weights across both branches)

```
Input: 105×105×3 image
  │
  ├─ Conv2D (64 filters, 10×10) → ReLU → MaxPool
  ├─ Conv2D (128 filters, 7×7)  → ReLU → MaxPool
  ├─ Conv2D (128 filters, 4×4)  → ReLU → MaxPool
  ├─ Conv2D (256 filters, 4×4)  → ReLU
  ├─ Flatten
  └─ Dense (4096 units, Sigmoid)
       ↓
  Embedding Vector
```

Both the **live image** and the **reference image** pass through this same network (shared weights = "Siamese"). Their embeddings are then compared using a custom L1 Distance Layer:

```python
class L1Dist(Layer):
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
```

The distance vector is passed to a final `Dense(1, sigmoid)` layer that outputs a **similarity score from 0 to 1**.

### Training Configuration

| Component | Choice | Reason |
|---|---|---|
| **Loss** | Binary Cross-Entropy | Pair labels are binary (same / different) |
| **Optimizer** | Adam | Fast convergence with adaptive learning rate |
| **Training Loop** | Custom `@tf.function` | Required to handle dual-input batch structure |

---

## 🔄 End-to-End Pipeline

```
1. DATA COLLECTION
   └─ Capture anchor + positive images via webcam (OpenCV)
   └─ Download LFW negatives from Kaggle

2. PREPROCESSING
   └─ Resize all images to 105×105 pixels
   └─ Normalize pixel values to [0, 1]
   └─ Build (anchor, pos/neg, label) triplets

3. TF DATA PIPELINE
   └─ Preprocessing map → Shuffle → Batch → Prefetch

4. MODEL BUILDING
   └─ CNN embedding subnetwork
   └─ Custom L1 Distance Keras layer
   └─ Final similarity classifier

5. TRAINING
   └─ Custom GradientTape training loop
   └─ Monitor binary cross-entropy convergence

6. EVALUATION
   └─ Threshold predictions at 0.5
   └─ Compute Precision, Recall, F1 Score

7. SAVE
   └─ model.save('siamesemodel.h5')

8. REAL-TIME VERIFICATION
   └─ Load model → Capture webcam frame
   └─ Run 50 comparisons against positive image pool
   └─ Apply detection + verification thresholds
   └─ Output: ✅ VERIFIED or ❌ NOT VERIFIED

9. KIVY APP
   └─ Load model at startup
   └─ Live webcam feed in GUI
   └─ One-button verification with on-screen result
```

---

## 📡 Verification Logic

The real-time verification mimics the logic behind **Apple Face ID** — multiple comparisons are run to reduce single-frame noise:

```python
def verify(model, detection_threshold=0.5, verification_threshold=0.6):
    results = []
    for image in os.listdir('application_data/verification_images'):
        input_img = preprocess('application_data/input_image/input_image.jpg')
        val_img   = preprocess(os.path.join('application_data/verification_images', image))
        result    = model.predict([input_img, val_img])
        results.append(result)

    detection    = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir('application_data/verification_images'))
    verified     = verification > verification_threshold

    return results, verified
```

| Threshold | Value | Role |
|---|---|---|
| **Detection** | 0.5 | Minimum confidence for a single comparison to count as a match |
| **Verification** | 0.6 | Minimum ratio of passed comparisons to confirm identity |

Both thresholds are tunable — in higher-security deployments, raising the verification threshold lowers the False Accept Rate at the cost of more false rejects.

---

## 🖥️ Kivy Desktop Application

The system ships as a **cross-platform desktop app** built with Kivy — no browser, no cloud dependency.

```
User opens app
    │
    ├─ Live webcam feed displayed
    │
    ├─ User presses [Verify]
    │
    ├─ 50 comparisons run against enrolled image pool
    │
    └─ Result displayed: ✅ VERIFIED or ❌ UNVERIFIED
```

Compatible with **Windows, macOS, Linux** — and extensible to Android/iOS via Kivy's mobile runtime.

---

## 📊 Performance Evaluation

| Metric | Description | Target in Biometric Context |
|---|---|---|
| **Precision** | Of all "verified" outputs, how many were correct? | High — minimize false accepts (security risk) |
| **Recall** | Of all genuine users, how many were correctly verified? | High — minimize false rejects (user friction) |
| **F1 Score** | Harmonic mean balancing both | Maximized |

> **Note:** In production biometric systems, **False Accept Rate (FAR)** is the critical failure mode — incorrectly granting access to an impostor. The dual-threshold design allows teams to tune this tradeoff explicitly based on deployment risk level.

---

## 🚀 Getting Started

### Prerequisites

Python 3.8+ installed on your system.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/thekavyaasharma/Face-Recognition-System.git
cd Face-Recognition-System

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow opencv-python kivy numpy matplotlib scikit-learn

# Optional: GPU acceleration
pip install tensorflow-gpu
```

### Run the Training Notebook

```bash
jupyter notebook FCR.ipynb
```

Run cells in order:
1. **Sections 1–2** — Data collection (webcam activates)
2. **Sections 3–4** — Preprocessing and model building
3. **Section 5** — Training
4. **Section 6** — Evaluation
5. **Section 7** — Real-time verification test
6. **Section 8** — Launch Kivy app

### Run the Application Directly

```bash
cd "Facial Recognition app"
python app.py
```

---

## 📁 Project Structure

```
Face-Recognition-System/
│
├── FCR.ipynb                        # Full pipeline: data → train → deploy
│
├── Facial Recognition app/
│   └── app.py                       # Kivy application entry point
│
├── data/
│   └── negative/                    # LFW negative samples
│
├── application_data/                # Created at runtime
│   ├── input_image/                 # Live webcam capture
│   └── verification_images/        # 50 enrolled positive images
│
├── siamesemodel.h5                  # Saved trained model
├── requirements.txt
├── siamese neural networks.pdf      # Koch et al. (2015) reference paper
└── README.md
```

---

## 🗺️ Roadmap

- [x] Webcam-based data collection pipeline
- [x] Image preprocessing and TF Data pipeline
- [x] Siamese Neural Network with custom L1 Distance layer
- [x] Custom training loop using `GradientTape`
- [x] Precision / Recall / F1 evaluation
- [x] Real-time verification function
- [x] Kivy cross-platform desktop app
- [ ] Multi-face detection in a single frame
- [ ] REST API endpoint for remote / cloud authentication
- [ ] Mobile deployment (Android/iOS via Kivy)
- [ ] Liveness detection (anti-spoofing)

---

## 📚 References

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese Neural Networks for One-shot Image Recognition.* ICML Deep Learning Workshop. *(included as `siamese neural networks.pdf`)*
- [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) — Kaggle
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Kivy Framework](https://kivy.org/)

---

## 👩‍💻 About the Author

**Kavya Sharma** is an AI/ML developer with a focus on building practical deep learning systems — from research implementation to end-to-end deployment.

VerifyNet demonstrates ownership across the full ML lifecycle: problem framing, custom architecture design, training infrastructure, evaluation methodology, and real-world deployment.

[![GitHub](https://img.shields.io/badge/GitHub-thekavyaasharma-181717?style=for-the-badge&logo=github)](https://github.com/thekavyaasharma)

---

*⭐ If this project was helpful or interesting, a star on GitHub is always appreciated!*
