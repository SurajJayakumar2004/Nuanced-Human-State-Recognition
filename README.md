# Real-Time Nuanced Human State Recognition

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS_Optimized-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-Video_Processing-5C3EE8)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

## 📌 Executive Summary
RHNS (Real-Time Nuanced Human State Recognition) v2.0 is a multimodal, Neuro-Symbolic Explainable AI (XAI) system. Moving beyond rudimentary facial emotion classification, RHNS is engineered to detect complex, contradictory human states such as **Sarcasm, Hiding Stress, Deep Focus, and Social Masking**. 

The core innovation of this system is **Cross-Modal Conflict Detection**. By utilizing a Gated Multimodal Fusion (GMF) architecture, the system mathematically calculates the congruence between a subject's visual expressions (via DINOv2) and their vocal prosody (via Wav2Vec 2.0). When the streams contradict, a Symbolic Rule Engine intercepts the neural prediction to output nuanced behavioral states.



---

## 🧠 System Architecture

RHNS v2.0 employs a **Decoupled Neuro-Symbolic Architecture**. It divides the classification task into two distinct layers: a Deep Learning Perception Engine and a Symbolic Reasoning Engine.

### 1. The Perception Engine (LSTM + GMF)
The neural network acts as the system's sensory cortex, trained exclusively on the 6 core ground-truth emotions (Happy, Sad, Angry, Fear, Disgust, Neutral) using a balanced subset of the CREMA-D dataset.
* **Visual Backbone:** DINOv2 extracts 384-D latent embeddings from isolated facial ROIs.
* **Audio Backbone:** Wav2Vec 2.0 extracts 768-D latent embeddings, supplemented by geometric prosody features (Jitter, Shimmer, Intensity).
* **Temporal Processing:** A 256-dimensional LSTM processes synchronized Audio-Visual chunks over a rolling 15-frame buffer (0.5s temporal window).



**The Gated Multimodal Fusion (GMF) Layer:**
Instead of simple concatenation, the fusion layer learns a dynamic sigmoid gate ($g$) to weigh the reliability of the incoming modalities:
$$g = \sigma(W_{gate} \cdot [h_{visual} \oplus h_{audio}] + b_{gate})$$
$$h_{fused} = g \cdot h_{visual} + (1 - g) \cdot h_{audio}$$
This allows the system to autonomously become "Audio Dominant" if the visual stream is obscured, or "Visual Dominant" in noisy environments.

### 2. The Symbolic Reasoning Engine (Master Overrides)
To detect states that lack pure, unacted datasets (like Sarcasm), RHNS utilizes a hard-coded geometric and behavioral rule engine based on MediaPipe Facial Action Units (FAUs) and Pose Landmarks.
* **Deception/Masking:** `IF Neural=Happy AND (FAU12 > 0.6 AND FAU6 < 0.2) -> "Fake / Polite Smile"`
* **Cognitive Load:** `IF Neural=Neutral AND Hand_To_Temple=True AND Gaze_Velocity < 0.2 -> "Deep Focus"`
* **Cross-Modal Sarcasm:** `IF Neural_Visual=Happy AND Neural_Audio=Angry -> "Sarcasm" (Conflict Score > 0.8)`

---

## 🖥️ Explainable AI (XAI) Dashboard

Standard AI models operate as "Black Boxes." RHNS is built as a transparent diagnostic tool. The 1280x720 "Clinical Cyber" UI exposes the neural network's internal math in real-time.

* **Visual Cortex (Left Panel):** Live telemetry of Lip Pullers (FAU12), Brow Furrows (FAU4), and Body Posture Matrix (Lean, Slump, Asymmetry).
* **Audio Cortex (Right Panel):** VU meters for Vocal Intensity and pitch Jitter.
* **Fusion Engine (Bottom Panel):** Exposes the GMF $g$ weight in real-time, proving whether the LSTM is currently attending to the Face or the Voice, alongside a Cross-Modal Conflict gauge.

---

## 🎮 Interactive Presentation Toggles ("God Mode")
To empirically prove the system's robustness during live demonstrations, the dashboard includes interactive pipeline manipulation toggles:

* **`r` (Rule Bypass):** Toggles the Symbolic Engine ON/OFF. Demonstrates how the system falls back to basic emotions (e.g., Boredom snaps to Neutral) when behavioral rules are disabled.
* **`v` (Visual Blindfold):** Zeroes out the visual tensor. Watch the GMF Gate dynamically slam to 100% Audio Dominance.
* **`a` (Audio Blindfold):** Zeroes out the audio tensor. Watch the GMF Gate dynamically slam to 100% Visual Dominance.
* **`l` (Landmark Mesh):** Overlays real-time, zero-latency MediaPipe geometric tessellation and skeletal pose lines onto the live video feed.
* **`c` (Camera ROI PIP):** Displays the isolated, normalized 112x112 grayscale face crop currently being digested by DINOv2.
* **`q` (Quit & Save):** Safely releases hardware locks and saves the session to an `.mp4` file.

---

## 🚀 Installation & Execution

### Prerequisites
* Python 3.11+
* PyTorch (MPS/CUDA supported)
* OpenCV, MediaPipe, Transformers, Torchaudio

### Setup
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
