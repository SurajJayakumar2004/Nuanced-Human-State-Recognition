# RHNS v2.0: Project Documentation Index

**Real-Time Nuanced Human State Recognition (RHNS v2.0)** is a multimodal affective computing system that detects base emotions and nuanced states (e.g., Sarcasm, Panic, Hiding Stress) from synchronized video and audio streams. The system combines a deep learning perception engine (DINOv2, Wav2Vec 2.0, LSTM) with a symbolic rule engine to achieve interpretable, real-time recognition on edge hardware.

---

## The Master Index

1. [Chapter 1: Abstract and Problem Statement](01_Abstract_and_Problem_Statement.md)  
   Details the limitations of 6-class emotion models, the cross-modal gap in affective computing, and introduces the neuro-symbolic approach.

2. [Chapter 2: System Architecture](02_System_Architecture.md)  
   Describes the Producer-Consumer multithreading model, the Deep Learning Perception Engine, and the Symbolic Reasoning Engine.

3. [Chapter 3: Dataset and Preprocessing](03_Dataset_and_Preprocessing.md)  
   Covers the CREMA-D dataset, class imbalance, the "Double-Dipping" collapse, and the WeightedRandomSampler fix.

4. [Chapter 4: Feature Extraction Pipeline](04_Feature_Extraction_Pipeline.md)  
   Explains the synchronization of 16 kHz audio and 30 FPS video, and the dual extraction of geometric features and latent embeddings.

5. [Chapter 5: Visual Expert (DINOv2)](05_Visual_Expert_DINOv2.md)  
   Details Meta's DINOv2 Vision Transformer as the visual backbone, patch embeddings, and the MPS fallback for Apple Silicon.

6. [Chapter 6: Audio Expert (Wav2Vec 2.0)](06_Audio_Expert_Wav2Vec2.md)  
   Details Meta's Wav2Vec 2.0 as the acoustic backbone, raw waveform processing, and prosody extraction (jitter, intensity).

7. [Chapter 7: Temporal Processing (LSTM)](07_Temporal_Processing_LSTM.md)  
   Explains the 15-frame rolling buffer, LSTM architecture, and micro-expression detection.

8. [Chapter 8: Gated Multimodal Fusion (GMF)](08_Gated_Multimodal_Fusion_GMF.md)  
   Covers the sigmoid gate, fusion equation, and Cross-Modal Conflict score.

9. [Chapter 9: Symbolic Reasoning Engine](09_Symbolic_Reasoning_Engine.md)  
   Describes the rule-based overrides in `utils/fusion.py` and the shift from pure connectionism to neuro-symbolic reasoning.

10. [Chapter 10: XAI Dashboard and Visualization](10_XAI_Dashboard_and_Visualization.md)  
    Details the segregated visual layout, terminal reasoning logs, and interactive presentation toggles.

11. [Chapter 11: Hardware Optimizations](11_Hardware_Optimizations.md)  
    Covers PyTorch MPS device mapping, threading locks, and DataLoader stability on macOS.

---

## Appendices & Resources

### Core Technologies Used

| Technology | Role |
|------------|------|
| **PyTorch** | Deep learning framework; MPS/CUDA acceleration |
| **OpenCV** | Video capture, rendering, image preprocessing |
| **MediaPipe** | Face Mesh, Pose, Hands for FAU and posture extraction |
| **DINOv2** | Meta's Vision Transformer for 384-D visual embeddings |
| **Wav2Vec 2.0** | Meta's self-supervised model for 768-D acoustic embeddings |

### Quick Reference

- **Entry point:** `main.py`
- **Models:** `models/fusion_head.py`, `models/visual_expert.py`, `models/audio_expert.py`
- **Rules:** `utils/fusion.py`
- **Training:** `scripts/train_temporal_model.py`
- **Feature extraction:** `scripts/extract_features.py`
