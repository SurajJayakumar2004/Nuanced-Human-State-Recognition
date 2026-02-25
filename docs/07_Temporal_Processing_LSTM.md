# 07 — Temporal Processing (LSTM)

## Chapter Overview

This document details the necessity of temporal modeling in affective computing, the mathematics of the Long Short-Term Memory (LSTM) network, and the specific 15-frame rolling buffer implemented in RHNS v2.0. It comprises three sections: (1) theoretical background on temporal dynamics of emotion; (2) practical implementation of the temporal pipeline; and (3) an FAQ addressing anticipated defense queries.

---

# Part I — Theoretical Background (Temporal Dynamics of Emotion)

## 1.1 The Static Frame Fallacy

### 1.1.1 Why Isolated Frames Fail

**Static frame analysis**—treating each video frame or audio snippet as an independent sample—is insufficient for nuanced state recognition. A single frame captures a snapshot of a dynamic process; it lacks the temporal context required to distinguish genuine expressions from transient artifacts or ambiguous configurations.

**Concrete example:** The **apex of a normal blink**—the moment when the eyelids are momentarily closed—can resemble a furrowed brow or lowered gaze when viewed in isolation. A classifier trained on static frames may misclassify this as "Sadness" or "Fatigue" because the closed-eye configuration shares visual features with those states. With **temporal context**—the frames before and after showing the blink's onset and offset—the system can recognize that the movement is a brief, involuntary blink and not an affective expression. The dynamics (rapid onset, rapid offset, duration ~100–150 ms) disambiguate the interpretation.

### 1.1.2 Micro-Expressions

A **micro-expression** is a brief, involuntary facial movement that reveals an emotion that the individual may be attempting to conceal or suppress. Micro-expressions typically last **40 to 500 ms** (0.04 to 0.5 seconds)—too fast for casual observation but detectable with frame-by-frame analysis. They are characterized by:

- **Short duration:** Often 1/25 to 1/5 of a second.
- **Low intensity:** Subtle muscle movements.
- **Involuntary:** Difficult to suppress or fake.

A system that processes only single frames or very short windows (e.g., 3 frames at 30 FPS = 100 ms) may miss micro-expressions that span 200–400 ms. A system with a 0.5 s window (15 frames at 30 FPS) can capture at least one complete micro-expression cycle.

---

## 1.2 Recurrent Neural Networks (RNNs) vs. LSTMs

### 1.2.1 Vanilla RNNs and the Vanishing Gradient Problem

A **vanilla RNN** processes a sequence by maintaining a hidden state $h_t$ that is updated at each time step:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

The hidden state is passed forward; the gradient is passed backward during training. For **long sequences** (e.g., 15 or more steps), the gradient must flow back through many time steps. The gradient is multiplied by the same weight matrix $W_{hh}$ at each step. If the largest eigenvalue of $W_{hh}$ is less than 1, the gradient **vanishes** exponentially; if greater than 1, it **explodes**. In practice, the gradient often vanishes, so the network cannot learn long-range dependencies. The RNN effectively "forgets" early frames when processing long video sequences.

### 1.2.2 The LSTM Solution: Cell State and Gating

The **LSTM** (Hochreiter & Schmidhuber, 1997) introduces a **cell state** $C_t$ that is updated via **gating mechanisms**. The cell state is a separate pathway from the hidden state; it is designed to allow gradients to flow **unchanged** over long sequences (in the ideal case). The **forget gate** controls how much of the previous cell state is retained; the **input gate** controls how much of the new candidate is added. This design mitigates the vanishing gradient problem and enables the network to learn long-range temporal dependencies.

---

## 1.3 LSTM Mathematics

### 1.3.1 Formal Definitions

At time step $t$, the LSTM cell receives the previous hidden state $h_{t-1}$, the previous cell state $C_{t-1}$, and the current input $x_t$. It computes:

**Forget Gate:**
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**Input Gate:**
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**Candidate Cell State:**
$$
\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

**Cell State Update:**
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**Output Gate:**
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**Hidden State:**
$$
h_t = o_t \odot \tanh(C_t)
$$

where $\sigma$ is the sigmoid function, $\odot$ denotes element-wise multiplication, and $[h_{t-1}, x_t]$ denotes concatenation of the previous hidden state and current input.

### 1.3.2 Interpretation

- **Forget gate $f_t$:** Controls how much of $C_{t-1}$ is retained. Values near 0 discard the past; values near 1 preserve it.
- **Input gate $i_t$:** Controls how much of the candidate $\tilde{C}_t$ is added to the cell state.
- **Cell state $C_t$:** The long-term memory; it accumulates information over time and can retain it across many steps when $f_t \approx 1$.
- **Output gate $o_t$:** Controls how much of the cell state is exposed in the hidden state $h_t$.

---

# Part II — Practical Implementation (The RHNS v2.0 Temporal Pipeline)

## 2.1 The 15-Frame Rolling Buffer

### 2.1.1 Sequence Length $L = 15$

The system uses a fixed sequence length **$L = 15$** frames. At 30 FPS, this yields:

$$
\Delta t = \frac{15}{30} = 0.5 \text{ seconds}
$$

of temporal context.

### 2.1.2 Why 15 Frames / 0.5 Seconds

**Lower bound:** Micro-expressions last 40–500 ms. A window shorter than ~200 ms may miss complete micro-expressions. Fifteen frames at 30 FPS (500 ms) covers the typical micro-expression duration.

**Upper bound:** Longer windows increase context but also **latency** and **memory**. For real-time inference, the system must process the buffer within the ~200 ms inference cycle. A 30-frame buffer (1 s) would double the sequence length and roughly double LSTM compute. A 60-frame buffer (2 s) would further increase latency and risk the system feeling sluggish. The 0.5 s window balances:

- **Sufficient context** for micro-expressions and short prosodic phrases.
- **Low latency** for real-time feedback.
- **Moderate compute** so the LSTM runs within the inference budget.

**Empirical:** 15 frames was chosen empirically; it provides sufficient temporal coverage for CREMA-D clips (typically 1–3 s) while keeping the model tractable. The rolling buffer is updated each inference cycle: the newest fused vector is appended, and the oldest is dropped (FIFO with maxlen=15).

---

## 2.2 LSTM Architecture in PyTorch

### 2.2.1 Implementation Location

The LSTM is implemented in **`models/fusion_head.py`** as part of the `NuancedStateClassifier` class. It is not in a separate `temporal_model.py`; the temporal processing is integrated into the fusion head.

### 2.2.2 Input Dimensionality

The LSTM receives the **sequence of fused multimodal vectors** $h_{\text{fused}}^{(1)}, \ldots, h_{\text{fused}}^{(15)}$, each of dimension **256** (the output of the GMF projection layer). The input shape is $(B, T, 256)$ where $B$ is batch size and $T = 15$.

### 2.2.3 Hidden Dimensionality: 128

The LSTM uses **hidden_size=128** and **num_layers=1**. The choice of 128 (rather than 256 or 512) is a **regularization** measure to prevent overfitting on the CREMA-D dataset. The model has ~6K trainable samples per class (with capping); a larger hidden size would increase parameter count and overfitting risk. The 128-D hidden state is sufficient to capture the temporal dynamics of the 256-D fused sequence while keeping the model compact.

### 2.2.4 PyTorch Configuration

```python
self.lstm = nn.LSTM(
    input_size=hidden_dim,      # 256
    hidden_size=lstm_hidden_size,  # 128
    num_layers=lstm_num_layers,    # 1
    batch_first=True,
)
```

---

## 2.3 Sequence-to-Vector Pooling

### 2.3.1 Why Only the Last Hidden State

The LSTM produces an output sequence $(h_1, h_2, \ldots, h_T)$ of shape $(B, T, 128)$. The **last hidden state** $h_T$ is used for classification:

$$
h_T = \text{lstm\_out}[:, -1, :] \in \mathbb{R}^{B \times 128}
$$

**Rationale:** The LSTM is designed so that the hidden state at each step encodes a **summary of the sequence up to that step**. The final hidden state $h_T$ thus encodes the entire sequence. It is a **compressed representation** of the 15-frame temporal dynamics. Using $h_T$ for classification is a form of **sequence-to-vector** pooling: the variable-length (or fixed-length) sequence is reduced to a single vector that captures the temporal context.

### 2.3.2 Alternative Pooling Strategies

Other strategies include **mean pooling** over time ($\bar{h} = \frac{1}{T}\sum_t h_t$) or **attention pooling**. The last hidden state is chosen because (1) it is the most recent and thus emphasizes recent frames, which may be more relevant for the current affective state; (2) it is computationally efficient (no extra aggregation); (3) it is standard for sequence classification in RNNs. For affective computing, the "current" state often depends most on recent expression; $h_T$ naturally emphasizes the end of the window.

### 2.3.3 Concatenation with Geometric Features

The last hidden state $h_T$ is concatenated with the **last-step geometric vector** (FAU12, FAU6, FAU4, jitter, intensity) of dimension 5:

$$
x = [h_T; \text{geo}] \in \mathbb{R}^{128 + 5 = 133}
$$

This vector is passed through an MLP (64→64→6) to produce the final 6-class logits.

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why use an LSTM for temporal processing instead of newer architectures like Temporal Convolutional Networks (TCNs) or a Time-Spatio Transformer?

### Brief

LSTMs are well-established for sequence modeling, have fewer parameters than Transformers for short sequences, and are sufficient for the 15-frame window. TCNs and Time-Spatio Transformers offer alternatives but add complexity; the LSTM choice balances performance, simplicity, and compatibility with the CREMA-D scale.

### Detailed

**TCNs:** Temporal Convolutional Networks use dilated causal convolutions to capture long-range dependencies. They are parallelizable (no sequential recurrence) and can be faster to train. However, they require careful design of receptive field size; for 15 frames, the LSTM's recurrence is already efficient. TCNs also lack the explicit "memory" mechanism of LSTMs; the LSTM's cell state provides a clear interpretable pathway for temporal accumulation.

**Time-Spatio Transformers:** Transformers apply self-attention over the sequence. They excel at long sequences and can model arbitrary pairwise dependencies. For a 15-frame sequence, the quadratic cost of attention ($O(T^2)$) is modest (225 pairs), but Transformers typically require more data and compute to train than LSTMs. The CREMA-D dataset has limited training samples; an LSTM with 128 hidden units is less prone to overfitting than a Transformer.

**LSTM choice:** The LSTM is a proven architecture for affective computing (e.g., emotion recognition from video). It handles the 15-frame sequence efficiently, has fewer parameters than a Transformer, and integrates well with the existing GMF pipeline. The choice is pragmatic: the LSTM meets the performance requirements without the complexity of newer architectures.

### Comprehensive

The choice reflects **task complexity**, **data scale**, and **engineering constraints**. **Task complexity:** For 15-frame sequences, the temporal structure is relatively short. LSTMs are designed for such sequences; the vanishing gradient is less severe over 15 steps. A Transformer would add global attention over 15 positions—potentially useful but not strictly necessary for the affective dynamics in this window. **Data scale:** CREMA-D provides ~6K training samples (with capping). Transformers typically require 10–100× more data to train effectively. An LSTM with 128 hidden units has on the order of 100K parameters for the recurrent layer; a small Transformer would have 500K–1M. The LSTM's smaller parameter count reduces overfitting risk. **Engineering constraints:** The system targets real-time inference on edge hardware (M-series Macs). LSTMs are well-optimized in PyTorch; MPS support is mature. Transformers may have different optimization profiles. The LSTM choice is thus a **conservative** one: it meets the requirements with proven, efficient technology. Future work could explore TCNs or Transformers if the sequence length or dataset scale increases.

---

## Q2: How does the system balance the trade-off between having a long enough temporal window to understand context, and a short enough window to remain "real-time"?

### Brief

The 15-frame window (0.5 s at 30 FPS) is chosen to capture micro-expressions (40–500 ms) without exceeding the inference budget. Longer windows would increase latency; shorter windows would miss temporal dynamics. The 0.5 s window is the minimum sufficient for the task.

### Detailed

**Context requirement:** Micro-expressions last 40–500 ms; prosodic phrases span 200–800 ms. A window shorter than ~200 ms may miss complete micro-expressions. A window of 0.5 s (15 frames) covers the typical micro-expression duration and at least one short prosodic phrase. This provides sufficient context for the LSTM to learn temporal dynamics.

**Latency constraint:** The inference thread runs at ~5 Hz, so each cycle has ~200 ms. The pipeline must complete within this budget: MediaPipe, DINOv2, Wav2Vec 2.0, GMF, LSTM, rules. The LSTM processes 15 steps; doubling to 30 would roughly double LSTM compute. The 15-frame window keeps the LSTM portion of the pipeline within ~5–15 ms, leaving the majority of the budget for the Transformers.

**Memory:** The rolling buffer holds 15 vectors of 256-D (fused) + 15 vectors of 768-D (audio) + 15 vectors of 384-D (visual) + 15 vectors of 5-D (geometric). Doubling the window would double memory. For edge deployment with unified memory, 15 frames is a reasonable footprint.

**Trade-off:** The system prioritizes **sufficiency** over **maximality**. The 0.5 s window is sufficient for micro-expressions and short prosodic context. It is not maximal—a 1 s or 2 s window might capture more context—but it is the minimum that meets the task requirement while preserving real-time performance.

### Comprehensive

The balance is a **Pareto optimization** over three dimensions: (1) **temporal coverage** (window long enough for micro-expressions and prosody), (2) **latency** (window short enough to keep inference within the cycle budget), and (3) **memory** (window short enough to fit in edge device memory). The 15-frame choice sits at the Pareto frontier: shorter windows (e.g., 7 frames) would reduce latency but risk missing micro-expressions; longer windows (e.g., 30 frames) would increase coverage but exceed the inference budget and memory. The 0.5 s window is also **aligned with human perception**: affective state changes are typically perceived on timescales of hundreds of milliseconds to seconds. A 0.5 s window matches the lower end of this range, providing near-instantaneous feedback while preserving temporal context. The design choice is thus **task-driven** (micro-expression duration) and **constraint-driven** (inference budget, memory).

---

## Q3: Mathematically, how does the Forget Gate ($f_t$) in the LSTM allow the network to ignore irrelevant past frames (like a prolonged period of silence)?

### Brief

The forget gate $f_t$ multiplies the previous cell state $C_{t-1}$ element-wise. When $f_t \approx 0$, the product $f_t \odot C_{t-1}$ is near zero, effectively discarding the past. When the input indicates irrelevance (e.g., silence), the network can learn to output low $f_t$, clearing the cell state.

### Detailed

The cell state update is:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

The forget gate $f_t \in [0, 1]^d$ (element-wise) controls how much of each dimension of $C_{t-1}$ is retained. If $f_t^{(i)} = 0$ for dimension $i$, then $C_t^{(i)} = 0 \cdot C_{t-1}^{(i)} + i_t^{(i)} \cdot \tilde{C}_t^{(i)} = i_t^{(i)} \cdot \tilde{C}_t^{(i)}$. The previous value $C_{t-1}^{(i)}$ is **completely discarded**. If $f_t^{(i)} = 1$, the previous value is fully retained (modulo the input gate's contribution).

**Prolonged silence:** When the input $x_t$ indicates silence (e.g., low audio energy, neutral face over many frames), the network can learn to output $f_t \approx 0$. The cell state is then "reset"—the accumulated past (e.g., a smile from 10 frames ago) is forgotten. The LSTM treats the silence as a **segment boundary** and starts fresh. When speech or expression resumes, $i_t$ and $\tilde{C}_t$ add new information; the cell state reflects only the recent, relevant content.

**Learning:** The forget gate weights $W_f$ and $b_f$ are learned during training. The network learns to output low $f_t$ when the input suggests that the past is no longer relevant (e.g., silence, neutral expression, or a transition to a new affective segment). This is a form of **learned attention over time**—the network decides when to retain and when to discard.

### Comprehensive

The forget gate implements **adaptive memory**: the network learns to retain or discard the past based on the current input. Formally, $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ is a **gated update**—the forget gate modulates the flow of information from the past. When $f_t \approx \mathbf{0}$, the update reduces to $C_t \approx i_t \odot \tilde{C}_t$; the cell state is dominated by the current input, and the past is effectively ignored. This is the mechanism by which the LSTM can "reset" during prolonged silence or neutral periods. The alternative—a vanilla RNN with no forget mechanism—would always accumulate the past; there would be no way to clear irrelevant history. The forget gate thus provides **selective memory**: the network retains what is relevant and discards what is not. For affective computing, this is critical: a 5-second silence in the middle of a clip should not influence the final prediction; the forget gate allows the LSTM to clear the pre-silence context and focus on the post-silence expression.
