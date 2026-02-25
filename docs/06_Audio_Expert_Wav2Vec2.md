# 06 — Audio Expert (Wav2Vec 2.0)

## Chapter Overview

This document details the use of Meta's **Wav2Vec 2.0** as the acoustic backbone of RHNS v2.0. It comprises three sections: (1) theoretical background on self-supervised speech representations; (2) practical implementation of the audio pipeline; and (3) an FAQ addressing anticipated defense queries.

---

# Part I — Theoretical Background (Self-Supervised Speech Representations)

## 1.1 Beyond Traditional Acoustics

### 1.1.1 Handcrafted Features: MFCCs and Beyond

Traditional speech and affective computing relied on **handcrafted acoustic features** derived from signal processing. **Mel-Frequency Cepstral Coefficients (MFCCs)** are the canonical example: the waveform is windowed, transformed to the frequency domain via FFT, passed through a mel filterbank (mimicking human auditory perception), and the log-magnitude spectrum is decorrelated via the discrete cosine transform (DCT). The result is a fixed-dimensional vector per frame (e.g., 13–39 coefficients) that captures spectral envelope and formant structure.

**Limitations:** MFCCs are designed for speech recognition and speaker identification. They emphasize phonetic content and suppress prosodic variation (pitch, intensity contours). For **affective computing**, prosody is critical—stress, emotion, and arousal manifest in pitch variability, intensity, and timing. MFCCs also discard phase information and assume a fixed mel scale; they are not optimized for emotion discrimination. Handcrafted features require domain expertise to design and may miss task-relevant structure.

### 1.1.2 The Paradigm Shift: Raw Waveform Processing

**Data-driven raw waveform processing** eliminates handcrafted feature design. The model receives the raw waveform (or a minimally preprocessed version) and learns a hierarchical representation end-to-end. Wav2Vec 2.0, HuBERT, and similar models demonstrate that raw waveforms contain sufficient information for the network to discover both phonetic and prosodic structure. The learned representations outperform handcrafted features on downstream tasks (ASR, emotion recognition) because they are optimized for the data distribution rather than human-designed heuristics.

---

## 1.2 The Wav2Vec 2.0 Architecture

### 1.2.1 Dual-Layer Design

Wav2Vec 2.0 (Baevski et al., 2020) comprises two principal components:

**1. CNN Feature Encoder**

A **multi-layer convolutional neural network** maps the raw waveform to a sequence of latent speech representations. The encoder uses 1D convolutions with a temporal stride that reduces the sequence length. For `facebook/wav2vec2-base`, the encoder produces one frame per ~20 ms of audio (depending on kernel sizes and strides). The output is a sequence of vectors in a continuous latent space—each vector represents a short segment of the waveform.

**2. Transformer Context Network**

A **Transformer encoder** processes the CNN output sequence. It applies self-attention across all time steps, building **contextualized representations** that incorporate information from the entire utterance. Each output vector $h_t$ depends on the full input sequence, not just the local CNN frame. This enables the model to capture long-range dependencies (e.g., prosodic contours, phrase-level stress patterns) that are essential for affective content.

### 1.2.2 Output

The Transformer produces a sequence of hidden states of shape $(T', 768)$, where $T'$ is the number of time steps (determined by the CNN's temporal reduction) and 768 is the hidden dimension. For downstream tasks, this sequence can be pooled (e.g., mean pooling) to produce a fixed-size embedding, or used as a sequence for temporal modeling.

---

## 1.3 Contrastive Predictive Coding

### 1.3.1 Self-Supervised Objective

Wav2Vec 2.0 is trained via **self-supervision** on unlabeled speech (e.g., LibriSpeech). The training objective combines:

1. **Masking:** Random spans of the CNN output are masked (replaced with a learnable mask embedding).
2. **Quantization:** A **quantization module** maps the continuous CNN output to discrete speech units via a Gumbel-Softmax distribution over learned codebooks. This discretization is used only during pretraining.
3. **Contrastive task:** The Transformer must predict the quantized representation of the masked positions. For each masked position, the model receives a set of candidates (the correct quantized code and distractors) and is trained to identify the correct one via a contrastive loss.

Conceptually, this is analogous to **BERT's masked language modeling** adapted for speech: the model learns to reconstruct masked content from context. The contrastive formulation avoids the need for a softmax over a huge vocabulary and stabilizes training.

### 1.3.2 Transfer to Downstream Tasks

After pretraining, the Transformer's contextualized representations are used as features for downstream tasks (ASR, emotion recognition). The model is typically frozen or fine-tuned on labeled data. For RHNS v2.0, the model is used as a **frozen feature extractor**: the 768-D hidden states are extracted and passed to the fusion head without fine-tuning.

---

# Part II — Practical Implementation (The RHNS v2.0 Audio Pipeline)

## 2.1 Audio Ingestion & Normalization

### 2.1.1 Sampling Rate: 16,000 Hz

The system enforces a **strict 16,000 Hz** sampling rate. This requirement is imposed by the Wav2Vec 2.0 pretraining setup: the model was trained on LibriSpeech, which is 16 kHz. The CNN feature encoder has fixed kernel sizes and strides that assume this sample rate. The temporal resolution of the output (one frame per ~20 ms) is tied to the input sample rate. Using a different rate (e.g., 8 kHz or 44.1 kHz) would alter the effective time scale and degrade performance.

**PyAudio configuration:** The AudioThread captures at `rate=16000`, `channels=1`, `format=paInt16`. Each chunk accumulates approximately 16,000 samples (~1 second).

### 2.1.2 Conversion to Floating-Point

Incoming audio arrives as **16-bit signed integer** samples in the range $[-32768, 32767]$. The `_to_float_audio()` method converts to float32 in $[-1, 1]$:

$$
x_n = \frac{\text{audio}_n}{32768}
$$

where $\text{audio}_n$ is the raw int16 sample. This normalization preserves the dynamic range and is the standard input format for Wav2Vec 2.0 (and most neural speech models).

### 2.1.3 Normalization Note

The implementation does **not** apply zero-mean, unit-variance normalization. Wav2Vec 2.0 was pretrained on raw waveforms in $[-1, 1]$; the model's internal layers expect inputs in this range. Additional normalization (e.g., per-chunk z-score) could distort the distribution and harm performance. The conversion to $[-1, 1]$ is sufficient.

---

## 2.2 Latent Vector Generation

### 2.2.1 Forward Pass

The normalized waveform tensor of shape $(1, T)$ is passed through the `Wav2Vec2Model`:

```python
tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
with torch.no_grad():
    out = self._wav2vec2(tensor)
```

The model returns an object with `last_hidden_state` of shape $(1, T', 768)$, where $T'$ is the sequence length after the CNN's temporal reduction.

### 2.2.2 Mean Pooling

To produce a **fixed-size** embedding from the variable-length sequence, the system applies **mean pooling** over the time dimension:

$$
h_{\text{audio}} = \frac{1}{T'} \sum_{t=1}^{T'} h_t
$$

where $h_t \in \mathbb{R}^{768}$ is the Transformer hidden state at time $t$. The result $h_{\text{audio}} \in \mathbb{R}^{768}$ is a single vector that summarizes the entire chunk. This vector is used for real-time inference when a single embedding per chunk is required.

### 2.2.3 Sequence Output (Offline)

For offline feature extraction (e.g., CREMA-D preprocessing), `get_latent_sequence()` produces a $(15, 768)$ tensor for temporal alignment with the visual stream. When $T' \geq 15$, the sequence is **uniformly sampled** to 15 steps. When $T' < 15$, **linear interpolation** is applied along the time dimension to produce 15 steps. This ensures a fixed $(15, 768)$ shape regardless of input length.

---

## 2.3 Integration with Prosody

### 2.3.1 Dual Representation

The Audio Expert provides two complementary representations:

1. **768-D latent embedding:** A dense, learned representation that captures **semantic tone**—phonetic content, prosodic contour, speaker characteristics, and affective cues that the Transformer has learned to encode. This is the primary input to the fusion head's audio projection and GMF.

2. **Explicit prosody features:** **Jitter** (pitch period variability) and **RMS Intensity** (loudness). These are **geometric**—computed by deterministic algorithms from the waveform. They capture **physical vocal strain** and arousal that may not be fully encoded in the latent (e.g., fine-grained jitter requires explicit F0 tracking).

### 2.3.2 Fusion Head Input

The LSTM receives:
- **Audio latent sequence:** $(15, 768)$ from Wav2Vec 2.0 (or mean-pooled 768-D tiled to 15 steps in the real-time rolling buffer).
- **Geometric sequence:** $(15, 5)$ where the last two dimensions per step are jitter and intensity (tiled from the chunk-level values).

The fusion head concatenates the last-step geometric vector with the LSTM hidden state. Thus the model has access to both **semantic tone** (from the 768-D latent) and **physical vocal strain** (from jitter and intensity). The rule engine uses jitter and intensity directly for overrides (e.g., Hiding Stress); the neural model learns to weight them when they are informative.

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why process raw audio waveforms with Wav2Vec 2.0 instead of using a standard CNN on audio spectrograms?

### Brief

Wav2Vec 2.0 learns representations end-to-end from raw waveforms, capturing both phonetic and prosodic structure without handcrafted feature design. A CNN on spectrograms requires a fixed time-frequency representation (e.g., mel spectrogram) that may discard phase information and prosodic detail. Wav2Vec 2.0's self-supervised pretraining on large-scale speech produces representations that transfer better to affective tasks.

### Detailed

**Spectrogram-based CNNs:** A mel spectrogram is a 2D representation (time × frequency) derived from the STFT. It emphasizes spectral envelope and suppresses phase. A CNN trained on spectrograms learns from this fixed representation. For emotion recognition, spectrograms can work, but they are optimized for human-designed mel scaling and frame rate. Prosodic features (pitch contour, micro-prosody) may be undersampled or lost in the mel compression.

**Wav2Vec 2.0:** Processes the raw waveform directly. The CNN encoder learns a task-relevant representation from the data; no handcrafted feature design is required. The model was pretrained on 53,000+ hours of speech (LibriSpeech and beyond) via self-supervision. The learned representations capture phonetic, prosodic, and speaker information in a unified 768-D space. For affective computing, this provides a strong inductive bias: the model has seen diverse speech and learned structure that generalizes to emotion.

**Empirical:** Self-supervised speech models (Wav2Vec 2.0, HuBERT) consistently outperform spectrogram-based baselines on emotion recognition benchmarks. The raw waveform approach avoids the information loss of spectrogram computation and benefits from large-scale pretraining.

### Comprehensive

The choice reflects **representation learning** and **transfer learning** principles. **Representation learning:** Handcrafted features (MFCCs, mel spectrograms) impose a fixed transformation. The model can only learn from what survives that transformation. Raw waveform models learn the transformation jointly with the task, allowing the network to discover representations that maximize downstream performance. **Transfer learning:** Wav2Vec 2.0 is pretrained on orders of magnitude more data than typical emotion datasets. Its representations are robust and general. A spectrogram CNN trained from scratch on CREMA-D would overfit; using Wav2Vec 2.0 as a frozen feature extractor leverages the pretrained knowledge. **Prosody:** Affective content is carried partly by prosody—pitch, intensity, timing. Spectrograms capture some of this (e.g., pitch as harmonic structure) but at reduced resolution. Wav2Vec 2.0's CNN + Transformer architecture operates at a finer temporal granularity and can learn prosody-relevant structure. The RHNS architecture supplements the latent with explicit jitter and intensity to ensure physical vocal strain is available to the rule engine.

---

## Q2: What is the mathematical necessity of forcing the audio input to exactly 16 kHz for this specific transformer?

### Brief

Wav2Vec 2.0 was pretrained on 16 kHz audio (LibriSpeech). The CNN feature encoder has fixed convolutional strides and kernel sizes that produce a specific temporal resolution (one frame per ~20 ms) when the input is 16 kHz. A different sample rate would alter the effective time scale and violate the model's training distribution.

### Detailed

**Architecture constraint:** The CNN encoder applies 1D convolutions with fixed stride $s$. For an input of length $T$ samples, the output has length $T' = T / s$ (approximately, depending on padding). The stride is designed so that at 16 kHz, each output frame corresponds to ~20 ms of audio. This aligns with phonetic and prosodic timescales (phonemes ~50–100 ms, syllables ~200 ms).

**Sample rate mismatch:** If the input were 8 kHz, the same number of samples would represent 2× the wall-clock duration. The CNN would produce fewer frames for the same real-time span, effectively downsampling the temporal resolution. If the input were 44.1 kHz, the opposite would occur: more frames for the same duration, potentially introducing redundancy and distribution shift. The model's weights were learned under the 16 kHz assumption; changing the rate changes the effective receptive field and time scale.

**Mathematical view:** The CNN can be viewed as a learned filterbank with a fixed temporal stride. The stride in samples is $s = f_s \cdot \Delta t$, where $f_s$ is the sample rate and $\Delta t$ is the desired frame interval (~20 ms). At 16 kHz, $s = 16000 \times 0.02 = 320$ samples per frame (approximately; the actual architecture may differ). At 8 kHz, the same stride would yield one frame per 40 ms—halving the temporal resolution. The model expects the former.

### Comprehensive

The 16 kHz requirement is a **training distribution** constraint. Wav2Vec 2.0 was trained on LibriSpeech, which is 16 kHz. The entire pipeline—CNN encoder, Transformer, and the learned representations—is calibrated to this rate. Using a different rate is equivalent to **distribution shift**: the model receives inputs from a different distribution than it was trained on. The degradation may be mild (e.g., 8 kHz resampled to 16 kHz) or severe (e.g., 44.1 kHz with no resampling). To ensure consistency, RHNS captures and processes all audio at 16 kHz. Resampling (e.g., 44.1 kHz → 16 kHz) would be possible but adds complexity and potential artifacts; the design choice is to capture at 16 kHz natively via PyAudio configuration.

---

## Q3: How does the system handle variable-length audio chunks to consistently produce a fixed-size 768-D embedding for the LSTM?

### Brief

For real-time inference, each chunk is ~1 second (16,000 samples). The Wav2Vec 2.0 output is a variable-length sequence $(1, T', 768)$. **Mean pooling** collapses the time dimension: $h_{\text{audio}} = \frac{1}{T'} \sum_{t=1}^{T'} h_t$, producing a fixed $(768,)$ vector. For offline extraction with sequence output, **uniform sampling** or **linear interpolation** reduces or expands the sequence to 15 steps.

### Detailed

**Real-time path:** The inference thread receives audio chunks of approximately 16,000 samples (~1 s). Small variations (e.g., 15,900 or 16,100 samples) occur due to buffering. Wav2Vec 2.0 accepts variable-length input; the Transformer output has shape $(1, T', 768)$ where $T'$ depends on the input length. To produce a fixed 768-D vector, the system applies **mean pooling**:

$$
h_{\text{audio}} = \frac{1}{T'} \sum_{t=1}^{T'} h_t \in \mathbb{R}^{768}
$$

This is invariant to $T'$: regardless of chunk length, the output is always 768-D. The LSTM's rolling buffer appends one 768-D vector per inference cycle; the sequence is built over time, not from a single chunk.

**Offline path:** For CREMA-D preprocessing, `get_latent_sequence()` produces $(15, 768)$ to align with the visual sequence. When $T' \geq 15$, the system **uniformly samples** 15 indices: $i_k = \lfloor k \cdot (T'-1) / 14 \rfloor$ for $k = 0, \ldots, 14$, and extracts $h_{i_k}$. When $T' < 15$, **linear interpolation** is applied along the time dimension: the sequence is treated as a function $h(t)$ for $t \in [0, 1]$, and 15 evenly spaced values are interpolated. Both strategies guarantee a fixed $(15, 768)$ output.

### Comprehensive

The design separates **chunk-level** and **sequence-level** handling. At the chunk level, variable length is acceptable: Wav2Vec 2.0 processes any length, and mean pooling produces a fixed 768-D summary. The LSTM does not receive variable-length sequences from a single chunk; it receives a **temporal sequence** built from multiple chunks (one per inference cycle). Each chunk contributes one 768-D vector to the rolling buffer. The sequence length (15) is fixed by the buffer size, not by the chunk length.

For offline extraction, each CREMA-D clip has variable duration (1–3 seconds). The Wav2Vec 2.0 output has variable $T'$. The sampling/interpolation step normalizes to 15 steps so that the fusion head always receives $(15, 768)$. This ensures consistent tensor shapes for batching and avoids padding or truncation logic in the model. The trade-off: sampling may drop information when $T' \gg 15$; interpolation may introduce smoothing when $T' \ll 15$. For affective recognition, the 15-step resolution is sufficient to capture prosodic contours over the clip duration.
