# 04 — Feature Extraction Pipeline

## Chapter Overview

This document details the synchronization of raw audio and video streams in RHNS v2.0 and the dual extraction of **geometric features** (explicit, interpretable) and **latent embeddings** (implicit, high-dimensional). It comprises three sections: (1) theoretical background on multimodal alignment and feature types; (2) practical implementation of the visual and audio pipelines; and (3) an FAQ addressing anticipated defense queries.

---

# Part I — Theoretical Background (Multimodal Alignment & Feature Types)

## 1.1 The Synchronization Problem

### 1.1.1 Differing Sampling Rates

In real-time affective computing, video and audio are captured by distinct hardware with different temporal characteristics:

- **Video:** Sampled at **30 FPS** (frames per second). Each frame represents a snapshot at $t = k/30$ seconds for $k = 0, 1, 2, \ldots$. The inter-frame interval is $\Delta t_{\text{video}} = 1/30 \approx 33.3$ ms.

- **Audio:** Sampled at **16,000 Hz**. Each sample represents a point in the continuous waveform at $t = n/16000$ seconds for $n = 0, 1, 2, \ldots$. The inter-sample interval is $\Delta t_{\text{audio}} = 1/16000 = 62.5$ μs.

The fundamental challenge is that **video and audio have no natural common timebase**. A single video frame spans ~533 audio samples; a 1-second audio chunk spans 30 video frames. To perform multimodal fusion, the system must establish a correspondence between visual and acoustic events—e.g., aligning a smile onset with a vocal intensity peak—despite the differing granularities.

### 1.1.2 Alignment Strategies

Two principal strategies exist:

1. **Upsample to a common grid:** Resample one modality to match the other. For example, interpolate audio features to 30 Hz to align with video frames. This introduces interpolation artifacts and assumes a smooth temporal evolution.

2. **Temporal windowing with paired buffers:** Maintain rolling buffers for each modality. At each inference cycle, pair the **latest** video frame(s) with the **latest** audio chunk(s) that overlap in wall-clock time. The buffers are filled by producer threads; the consumer (inference thread) reads both under lock and treats them as co-temporal. This approach accepts **approximate** alignment—the frame and chunk are from the same inference interval—rather than sample-level precision.

RHNS v2.0 adopts the second strategy: lock-protected shared buffers, with the inference thread pairing the most recent frame and audio chunk at each cycle.

---

## 1.2 Temporal Windowing

### 1.2.1 The Rolling Buffer

A **temporal window** (or rolling buffer) holds the last $T$ samples from a stream. For RHNS, $T = 15$ (SEQ_LEN). The buffer is updated each inference cycle: the newest sample is appended, and the oldest is dropped (FIFO with fixed capacity).

### 1.2.2 Why 15 Frames / ~0.5 Seconds

- **Micro-expressions:** Brief, involuntary facial movements last 40–500 ms. A 15-frame window at 30 FPS spans 500 ms, capturing at least one micro-expression cycle.

- **Transient vocal prosody:** Pitch and intensity contours evolve over hundreds of milliseconds. A 0.5 s window captures a prosodic phrase or stress pattern.

- **Computational trade-off:** Longer windows increase context but also latency and memory. Fifteen frames balance temporal coverage with real-time constraints.

### 1.2.3 Sequence Construction

For the neural model, the rolling buffers produce fixed-length sequences:
- **Visual:** $(15, 384)$ — 15 frames of DINOv2 embeddings.
- **Audio:** $(15, 768)$ — 15 steps of Wav2Vec 2.0 embeddings (each step may be from a 1 s chunk, downsampled or interpolated to 15 steps).
- **Geometric:** $(15, 5)$ — 15 steps of [FAU12, FAU6, FAU4, jitter, intensity].

When the buffer is not yet full, zero-padding is applied from the start of the sequence.

---

## 1.3 Geometric vs. Latent Features

### 1.3.1 Geometric Features (Explicit, Deterministic)

**Geometric features** are computed by **explicit, rule-based algorithms** from raw sensor data. They are:

- **Interpretable:** Each dimension has a clear meaning (e.g., FAU12 = lip corner displacement, jitter = pitch variability).
- **Deterministic:** Given the same input, the same output is produced; no learned parameters.
- **Low-dimensional:** Typically a few to a few dozen dimensions.

Examples in RHNS:
- **FAU12 (Lip Corner Puller):** Intensity of smile, from Face Mesh landmark geometry.
- **FAU6 (Cheek Raiser):** Duchenne marker, from cheek region landmarks.
- **FAU4 (Brow Lowerer):** Furrowing, from brow landmarks.
- **Posture metrics:** Shoulder asymmetry, lean, slump, from Pose landmarks.
- **Vocal jitter:** Pitch period variability, from F0 track.
- **Intensity:** RMS energy of the audio chunk.

These features feed the **Symbolic Reasoning Engine** and support interpretable overrides (e.g., "high jitter + neutral face → Hiding Stress").

### 1.3.2 Latent Embeddings (Implicit, High-Dimensional)

**Latent embeddings** are produced by **pretrained deep neural networks**. They are:

- **Implicit:** The dimensions do not have direct semantic labels; they encode learned representations.
- **High-dimensional:** DINOv2 outputs 384-D; Wav2Vec 2.0 outputs 768-D.
- **Data-driven:** Learned from large-scale unsupervised or supervised training.

Examples in RHNS:
- **DINOv2 (384-D):** Visual embedding of the cropped face region. Captures appearance, expression, and context in a compact vector.
- **Wav2Vec 2.0 (768-D):** Acoustic embedding of the waveform. Captures phonetic, prosodic, and speaker characteristics.

These embeddings feed the **neural fusion head** (GMF, LSTM) for base-emotion classification. They capture information that geometric features cannot explicitly encode.

### 1.3.3 Complementary Roles

Geometric and latent features serve **complementary** roles: geometric features support **reasoning** (rules, interpretability); latent features support **perception** (learned classification). The fusion head concatenates the last-step geometric vector with the LSTM hidden state, allowing the model to combine both.

---

# Part II — Practical Implementation (The RHNS v2.0 Pipeline)

## 2.1 Visual Stream Extraction

### 2.1.1 OpenCV Frame Capture

The **CameraThread** uses **OpenCV** (`cv2.VideoCapture`) to capture frames from the default webcam at **640×480** and **30 FPS**. Each iteration:

1. Call `cap.read()` to obtain the latest frame (BGR format).
2. Acquire `frame_lock`.
3. Write `frame.copy()` to `frame_container["frame"]`.
4. Release `frame_lock`.

OpenCV handles the camera driver interface and provides a synchronous read; the Producer-Consumer design ensures that slow inference does not block capture.

### 2.1.2 MediaPipe Pipeline

The **VisualExpert** (`models/visual_expert.py`) runs a MediaPipe pipeline on each frame to extract geometric and latent features.

#### Face Mesh → FAUs

MediaPipe **Face Mesh** detects 468 facial landmarks. From these, the system computes three Facial Action Units via landmark geometry and intensity calculators:

| FAU | FACS Name | Description | Use |
|-----|-----------|-------------|-----|
| **FAU12** | Lip Corner Puller | Displacement of lip corners (smile) | Smile detection; low FAU12 + Happy voice → sarcasm cue |
| **FAU6** | Cheek Raiser | Duchenne marker, cheek elevation | Genuine vs. polite smile |
| **FAU4** | Brow Lowerer | Brow depression | Anger, concentration, frustration |

These are normalized to $[0, 1]$ and passed to the fusion head and rule engine.

#### Pose → Body Posture

MediaPipe **Pose** detects 33 body landmarks. From these, the system computes:

- **Shoulder Asymmetry:** $|y_{11} - y_{12}| > 0.05$ (normalized coordinates), where landmarks 11 and 12 are left and right shoulders. Asymmetry indicates postural tension or contempt-related head tilt.

- **Lean:** Forward ($z < -0.1$), back ($z > 0.1$), or neutral, from shoulder midpoint depth. Forward lean can indicate aggression or engagement; back lean can indicate disgust or withdrawal.

- **Posture Slump:** Nose-to-shoulder distance increases vs. a rolling baseline. Slump indicates low energy, boredom, or depression.

- **Shoulders Raised:** Shoulders unnaturally close to nose on the Y-axis. Indicates tension, anxiety, or panic.

- **Shoulder Rigidity:** Low variance of Y-distance between shoulders over time. High rigidity indicates controlled tension.

#### Hands → Gestures

MediaPipe **Hands** detects 21 landmarks per hand. The system computes:

- **Self-Touching (Hand-to-Face):** Hand landmarks near face zones (chin, mouth, temple). Indicates anxiety, boredom, or masking.

- **Finger Tapping:** High-frequency movement of fingertips. Indicates restlessness, controlled annoyance, or stress.

### 2.1.3 DINOv2 Latent Extraction

After Face Mesh identifies the face region, the face is cropped (with 20% margin), resized to 224×224, normalized (ImageNet), and passed through **DINOv2 ViT-S/14**. The CLS token output is a **384-D** embedding. This latent is appended to the visual sequence buffer for the fusion head.

---

## 2.2 Audio Stream Extraction

### 2.2.1 PyAudio Capture

The **AudioThread** uses **PyAudio** to capture **16 kHz mono** audio in **~1-second chunks**. Each iteration:

1. Read `chunk_size` (1024) samples at a time until 16,000 samples are accumulated.
2. Concatenate into a single buffer.
3. Acquire `audio_lock`.
4. Write the buffer to `audio_container["chunk"]` with `sample_rate=16000`.
5. Release `audio_lock`.

The format is `paInt16` (16-bit signed integer). The inference thread converts to float32 in $[-1, 1]$ for processing.

### 2.2.2 Acoustic Prosody: Intensity (RMS)

**Voice Intensity** is computed as the RMS (root-mean-square) energy of the audio chunk:

$$
\text{intensity} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x_n^2}
$$

where $x_n \in [-1, 1]$ are the normalized samples. The result is clamped to $[0, 1]$. High intensity indicates loud speech; low intensity may indicate whispered or subdued affect.

### 2.2.3 Acoustic Prosody: Vocal Jitter

**Vocal Jitter** quantifies **pitch period variability**—the instability of consecutive pitch periods. It is a well-established indicator of **physiological stress**, neurological conditions, and emotional arousal. When the vocal folds are under stress or tension, the period-to-period consistency decreases, and jitter increases.

**Standard formula (absolute jitter, consecutive periods):**

$$
\text{Jitter}_{\text{abs}} = \frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i-1}|
$$

where $T_i = 1/f_i$ is the pitch period (in seconds) of the $i$-th voiced frame, and $f_i$ is the fundamental frequency (F0) in Hz. Only voiced frames ($f_i > 0$) are used.

**RHNS implementation (Pitch Perturbation Quotient, PPQ):**

The AudioExpert uses a variant that normalizes by the mean period:

$$
\text{Jitter}_{\text{PPQ}} = \frac{\frac{1}{K} \sum_{i=1}^{K} |T_i - \bar{T}|}{\bar{T}}, \quad \bar{T} = \frac{1}{K} \sum_{i=1}^{K} T_i
$$

where $K$ is the number of voiced frames. This yields a dimensionless ratio in $[0, 1]$ (after clamping), suitable for fusion with other normalized features. High jitter ($> 0.6$) triggers the "Hiding Stress" rule when combined with a neutral face and rigid posture.

**F0 extraction:** The system uses `librosa.pyin()` (or `piptrack` as fallback) to obtain a frame-wise F0 track. Voiced frames are identified; unvoiced frames are excluded from the jitter computation.

### 2.2.4 Wav2Vec 2.0 Latent Extraction

The audio chunk is passed through **Wav2Vec 2.0 base**, producing a sequence of hidden states. For real-time inference, the mean-pooled 768-D vector is used per chunk; for offline extraction, `get_latent_sequence()` samples or interpolates to 15 steps for temporal alignment with the visual stream.

---

## 2.3 The Alignment Buffer

### 2.3.1 Thread-Safe Shared Containers

RHNS v2.0 uses **lock-protected dictionaries** (not `queue.Queue`) for inter-thread communication:

| Container | Lock | Producer | Consumer |
|-----------|------|----------|----------|
| `frame_container` | `frame_lock` | CameraThread | InferenceThread |
| `audio_container` | `audio_lock` | AudioThread | InferenceThread |
| `state_container` | `state_lock` | InferenceThread | Main loop (UI) |

Each container holds the **latest** value; older values are overwritten. This "mailbox" semantics ensures that the consumer always sees the most recent data without blocking producers.

### 2.3.2 Pairing and Sequence Construction

At each inference cycle (~0.2 s interval):

1. **Acquire locks and copy:** The InferenceThread acquires `frame_lock` and `audio_lock`, copies the latest frame and audio chunk, and releases the locks immediately.

2. **Extract features:** Run VisualExpert and AudioExpert on the copied data. Obtain: `visual_latent` (384-D), FAU dict, body_data, `audio_latent` (768-D), jitter, intensity.

3. **Append to rolling buffers:** Append to `_seq_visual`, `_seq_audio`, `_seq_geometric` (each `deque(maxlen=15)`).

4. **Build sequences:** Stack the buffer contents into `(15, 384)`, `(15, 768)`, `(15, 5)`. Zero-pad from the start if the buffer is not yet full.

5. **Pass to classifier:** The sequences are passed to `NuancedStateClassifier.predict()`.

### 2.3.3 Temporal Correspondence

The $i$-th frame in the visual sequence and the $i$-th step in the audio sequence are from the **same inference cycle** (or the same wall-clock window). Over 15 cycles, the buffers hold the last 15 such pairs. The alignment is **approximate**—the audio chunk is ~1 s and the video frame is instantaneous—but for affective recognition, this coarse alignment is sufficient for capturing prosody and expression over a ~0.5–2 s window.

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why does the system extract both explicit geometric features (FAUs/Pose) and implicit latent embeddings (DINOv2/Wav2Vec2)?

### Brief

Geometric features support **interpretable reasoning** (rules, overrides) and are required for the Symbolic Engine. Latent embeddings support **learned perception** (base-emotion classification) and capture information that geometric features cannot explicitly encode. Both are needed for the neuro-symbolic architecture.

### Detailed

**Geometric features** are used by the rule engine to override neural predictions when specific conditions are met. For example, "Neutral + jitter > 0.6 + rigidity > 0.85 → Hiding Stress" requires jitter, rigidity, and FAU values. These must be derived from explicit, deterministic algorithms so that the rules can reference them. The rule engine cannot operate on raw 384-D or 768-D vectors—it needs interpretable scalars and booleans.

**Latent embeddings** are used by the neural fusion head to learn base-emotion boundaries from data. DINOv2 and Wav2Vec 2.0 encode rich, high-dimensional representations that capture subtle variations in appearance and prosody. A rule-based system cannot enumerate all such variations; the neural model learns them from CREMA-D. The geometric features alone (3 FAUs + 2 prosody + 5 posture flags) are insufficient for robust 6-class classification—they lack the discriminative power of 384-D and 768-D embeddings.

**Combined:** The fusion head concatenates the LSTM last hidden state with the last-step geometric vector. The model thus learns to weight geometric cues (e.g., high jitter) when they are informative, while relying on latents for the base classification. The rule engine uses only geometric features for overrides. The two feature types serve complementary roles in perception and reasoning.

### Comprehensive

The dual extraction reflects the **division of labor** between data-driven and knowledge-driven components. **Perception** (mapping raw signals to base emotions) benefits from high-dimensional latent representations learned from large-scale data. **Reasoning** (mapping base emotions + context to nuanced states) benefits from explicit rules that reference interpretable geometric features. If the system used only geometric features, it would lack the representational capacity for subtle emotion discrimination. If it used only latents, the rule engine would have no interpretable inputs to condition on. The architecture therefore requires both: latents for the neural classifier, and geometric features for the symbolic overrides. The fusion head's concatenation of geometric with LSTM output allows the neural model to learn when to trust geometric cues (e.g., in high-conflict scenarios) while still leveraging the full latent representation for the primary classification task.

---

## Q2: How does the system handle the mathematical synchronization of a 30 FPS video stream with a 16 kHz audio stream?

### Brief

The system does not perform sample-level synchronization. Instead, it uses **temporal windowing** with paired buffers: at each inference cycle, the latest video frame and the latest audio chunk are treated as co-temporal. Both are appended to their respective 15-step rolling buffers, and the resulting sequences are passed to the fusion head. Alignment is approximate but sufficient for affective recognition.

### Detailed

**Frame rate:** Video at 30 FPS yields one frame every 33.3 ms. The inference thread runs at ~5 Hz, so it processes every 6th frame (approximately). The rolling buffer holds the last 15 frames from the last 15 inference cycles—roughly 0.5 s of video at 30 FPS.

**Audio rate:** Audio at 16 kHz is captured in ~1 s chunks. Each chunk contains 16,000 samples. The chunk is processed as a single unit: Wav2Vec 2.0 produces a sequence of hidden states; the system extracts a 768-D vector (mean-pooled or sampled to 15 steps for offline). For real-time, one 768-D vector per chunk is appended to the audio rolling buffer.

**Pairing:** At cycle $k$, the inference thread reads frame $f_k$ and chunk $a_k$. These are from overlapping wall-clock intervals: $f_k$ is from the most recent camera read, and $a_k$ is from the most recent 1 s of audio. The chunk $a_k$ may overlap with frames $f_{k-5}, \ldots, f_{k+5}$ (since 1 s of audio spans ~30 frames). The system does not align them sample-by-sample; it assumes that the affective state is relatively stable over the window and that the frame and chunk are representative of the same temporal segment.

**Sequence construction:** The visual sequence is built from 15 consecutive frames (one per inference cycle). The audio sequence is built from 15 consecutive chunks (one per inference cycle). The $i$-th visual step and $i$-th audio step are from the same cycle, so they are temporally aligned at the cycle level.

### Comprehensive

The synchronization strategy is **coarse temporal alignment** rather than **fine-grained sample alignment**. For affective computing, the relevant temporal scale is hundreds of milliseconds—micro-expressions (40–500 ms), prosodic phrases (200–800 ms), and emotional transitions (1–2 s). Sample-level alignment (e.g., interpolating audio to 30 Hz to match each frame) would add complexity without significant benefit, because the neural model and rule engine operate on aggregate features over windows. The key invariant is that the 15-frame visual buffer and the 15-step audio buffer both represent the last ~0.5–2 s of input, and the $i$-th step in each corresponds to the same inference cycle. This design is consistent with the Producer-Consumer architecture: each modality is captured independently at its natural rate, and the inference thread pairs them at the coarse granularity of its own cycle rate. The system accepts that the audio chunk is longer than the inter-frame interval; the Wav2Vec 2.0 output is a summary of the chunk, and the fusion head learns to combine it with the visual summary of the frame.

---

## Q3: What is Vocal Jitter, and how is it calculated mathematically to indicate physiological stress?

### Brief

Vocal jitter is the **variability of consecutive pitch periods** in voiced speech. It is computed from an F0 track: higher jitter indicates less stable pitch, which correlates with physiological stress, autonomic arousal, and neurological conditions. The standard formula uses the mean absolute difference between consecutive periods; RHNS uses a PPQ variant normalized by mean period.

### Detailed

**Definition:** In voiced speech, the vocal folds oscillate with a fundamental frequency $f_0$. The **pitch period** is $T = 1/f_0$ (seconds). In healthy, relaxed speech, $T$ is relatively stable from cycle to cycle. Under stress, the period becomes more variable—**jitter** quantifies this variability.

**Standard formula (absolute jitter, consecutive periods):**

$$
\text{Jitter}_{\text{abs}} = \frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i-1}|
$$

where $T_i = 1/f_i$ for voiced frame $i$, and $N$ is the number of voiced frames. The units are seconds (or milliseconds). This measure is sensitive to local period-to-period changes.

**Normalized variant (PPQ):** To make jitter comparable across speakers and pitch ranges, it is often normalized by the mean period:

$$
\text{Jitter}_{\text{PPQ}} = \frac{\frac{1}{K} \sum_{i=1}^{K} |T_i - \bar{T}|}{\bar{T}}, \quad \bar{T} = \frac{1}{K} \sum_{i=1}^{K} T_i
$$

This yields a dimensionless ratio. RHNS uses this PPQ variant, clamped to $[0, 1]$, for fusion with other normalized features.

**Physiological basis:** Jitter increases with (1) **autonomic arousal**—stress activates the sympathetic nervous system, affecting laryngeal muscle tension; (2) **neurological conditions**—e.g., Parkinson's disease, essential tremor; (3) **emotional arousal**—anxiety, fear, or suppressed anger can manifest as vocal tremor. In the context of "Hiding Stress," a neutral face with high jitter suggests that the voice is leaking stress that the face is masking.

### Comprehensive

Jitter is a well-validated **paralinguistic** measure in speech pathology and affective computing. The mathematical definition is grounded in the physics of vocal fold oscillation: the period $T$ is the time between successive glottal closures. Variability in $T$ reflects variability in the biomechanical control of the vocal folds. The choice of formula—absolute vs. PPQ—affects the scale and sensitivity. Absolute jitter is in seconds and is sensitive to pitch (higher pitch → shorter periods → potentially smaller absolute differences). PPQ normalizes by mean period, making it comparable across speakers. RHNS uses PPQ and clamps to $[0, 1]$ so that jitter can be used alongside other normalized features (FAU, intensity) in the fusion head and rule engine. The threshold of 0.6 for "Hiding Stress" was chosen empirically: values above this indicate clinically significant pitch instability that, when combined with a neutral face and rigid posture, triggers the override. The F0 track is obtained via `librosa.pyin()`, which uses a probabilistic YIN algorithm for robust pitch estimation in noisy or low-energy segments.
