# 08 — Gated Multimodal Fusion (GMF)

## Chapter Overview

This document details the core mathematical innovation of RHNS v2.0: the **Gated Multimodal Fusion (GMF)** layer and the **Cross-Modal Conflict** detector. It comprises three sections: (1) theoretical background on multimodal fusion strategies; (2) mathematical formulation and implementation; and (3) an FAQ addressing anticipated defense queries.

---

# Part I — Theoretical Background (Multimodal Fusion Strategies)

## 1.1 Early vs. Late Fusion

### 1.1.1 Early Fusion

**Early fusion** concatenates raw or low-level features from each modality before any high-level processing. For audio-visual affect recognition, this might mean concatenating mel spectrogram frames with image patches, or concatenating DINOv2 and Wav2Vec 2.0 outputs before classification.

**Failure mode — Dimensionality mismatch:** DINOv2 produces 384-D vectors; Wav2Vec 2.0 produces 768-D vectors (or sequences thereof). Concatenation yields 1152-D vectors. The mismatch is not only in dimension but in **semantic structure**: the two modalities encode different types of information (appearance vs. prosody) in incompatible coordinate systems. A single linear layer must learn to reconcile 384 visual dimensions with 768 audio dimensions, increasing parameter count and optimization difficulty. More critically, when modalities have different temporal resolutions (e.g., 15 visual frames vs. variable-length audio), early fusion requires explicit alignment that may introduce artifacts.

### 1.1.2 Late Fusion

**Late fusion** trains separate unimodal classifiers and combines their outputs (e.g., by averaging logits or probabilities). Each modality is processed independently; the fusion occurs only at the decision level.

**Failure mode — Ignored cross-modal correlations:** Late fusion assumes that the modalities contribute **independently** to the final prediction. It cannot model **interactions**—e.g., that a smile (visual) paired with flat prosody (audio) indicates sarcasm. The correlation between modalities is lost; the system cannot detect when they agree or disagree. For nuanced states that arise from cross-modal contradiction, late fusion is fundamentally limited.

### 1.1.3 Intermediate Fusion

**Intermediate fusion** combines modality-specific representations at an intermediate layer, before the final classifier. The GMF layer is an instance of intermediate fusion: it operates on projected latent vectors and produces a fused representation that the LSTM and classifier consume. This allows the model to learn cross-modal interactions while avoiding the raw concatenation of incompatible dimensions.

---

## 1.2 The Concatenation Fallacy

### 1.2.1 Simple Latent Concatenation

A naive approach is to concatenate the latent vectors:

$$
h_{\text{concat}} = [h_{\text{visual}} \oplus h_{\text{audio}}] \in \mathbb{R}^{384 + 768} = \mathbb{R}^{1152}
$$

and pass $h_{\text{concat}}$ to a classifier. This forces the network to process both modalities as a single flat vector.

### 1.2.2 Sub-Optimality in Noisy Environments

When one modality is **unreliable or absent**—e.g., the camera is blinded, the face is occluded, or the microphone is muted—the concatenated vector contains **dead or noisy dimensions**. If the visual stream is blinded, $h_{\text{visual}}$ may be zeros or random noise. The concatenated vector still has 384 visual dimensions; the network must learn to ignore them. There is no mechanism to **down-weight** the unreliable modality; the classifier receives 50% (or more) dead noise and must compensate through learned weights. This wastes capacity, increases overfitting risk, and degrades robustness.

### 1.2.3 The Gate as a Solution

A **learnable gate** $g \in [0, 1]$ dynamically adjusts the contribution of each modality. When the visual stream is blinded, the gate can learn to set $g \approx 0$, effectively routing the fused representation toward the audio stream. The model does not process dead noise; it **suppresses** the unreliable modality. This is the core advantage of gated fusion over concatenation.

---

## 1.3 The Attention-Based Gate Mechanism

### 1.3.1 Concept

A **learnable sigmoid gate** $g$ routes attention between modalities based on their **relative information density**. The gate is computed from both modalities—it "looks at" $h_{\text{visual}}$ and $h_{\text{audio}}$ and decides how much to trust each. When the visual stream is informative (clear face, strong expression), $g$ may be high; when the audio stream is informative (clear speech, distinctive prosody), $g$ may be low. The gate adapts **per sample** and **per time step**, allowing the model to handle varying conditions (lighting, occlusion, noise) without manual intervention.

### 1.3.2 Relation to Attention

The gate is a simplified form of **attention**: instead of a full attention distribution over many positions, we have a scalar that weights two "positions" (visual and audio). The sigmoid ensures $g \in [0, 1]$, and the convex combination $h_{\text{fused}} = g \cdot h_{\text{visual}} + (1-g) \cdot h_{\text{audio}}$ is a soft selection between the two streams. This design is computationally efficient and interpretable: $g$ can be displayed in the XAI Dashboard as "Visual vs. Audio dominance."

---

# Part II — Mathematical Formulation & Implementation

## 2.1 Dimensionality Alignment

### 2.1.1 Prerequisite

Before fusion, DINOv2 (384-D) and Wav2Vec 2.0 (768-D) outputs must be projected to a **shared dense dimension** $d = 256$. This alignment is implemented in `models/fusion_head.py` via linear transformations:

$$
h_{\text{visual}} = W_{\text{vis}} \, x_{\text{vis}} + b_{\text{vis}}, \quad h_{\text{audio}} = W_{\text{aud}} \, x_{\text{aud}} + b_{\text{aud}}
$$

where:
- $x_{\text{vis}} \in \mathbb{R}^{384}$, $W_{\text{vis}} \in \mathbb{R}^{256 \times 384}$, $b_{\text{vis}} \in \mathbb{R}^{256}$
- $x_{\text{aud}} \in \mathbb{R}^{768}$, $W_{\text{aud}} \in \mathbb{R}^{256 \times 768}$, $b_{\text{aud}} \in \mathbb{R}^{256}$

Thus $h_{\text{visual}}, h_{\text{audio}} \in \mathbb{R}^{256}$. The shared dimension enables a symmetric treatment of both modalities in the gate and fusion equations.

### 2.1.2 Why 256?

The choice $d = 256$ balances expressiveness and efficiency. It is large enough to preserve task-relevant information from both 384-D and 768-D inputs, while keeping the gate input (512-D) and LSTM input (256-D) manageable. The projection also serves as a **bottleneck**, encouraging the model to extract the most discriminative features from each modality.

---

## 2.2 The Gating Equation

### 2.2.1 Gate Computation

The gate scalar $g \in [0, 1]$ is computed as:

$$
g = \sigma\left( W_{\text{gate}} \cdot [h_{\text{visual}} \oplus h_{\text{audio}}] + b_{\text{gate}} \right)
$$

where:
- $[h_{\text{visual}} \oplus h_{\text{audio}}] \in \mathbb{R}^{512}$ is the concatenation of the two projected vectors
- $W_{\text{gate}} \in \mathbb{R}^{1 \times 512}$, $b_{\text{gate}} \in \mathbb{R}$
- $\sigma$ is the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Interpretation:** $g = 1$ → model relies entirely on visual; $g = 0$ → entirely on audio; $g = 0.5$ → equal weighting.

---

## 2.3 The Fusion Equation

### 2.3.1 Convex Combination

The fused vector passed to the LSTM is a **convex combination** of the projected modalities:

$$
h_{\text{fused}} = g \cdot h_{\text{visual}} + (1 - g) \cdot h_{\text{audio}}
$$

Since $g \in [0, 1]$, $h_{\text{fused}}$ lies on the line segment between $h_{\text{visual}}$ and $h_{\text{audio}}$ in $\mathbb{R}^{256}$. The gate $g$ controls the interpolation: when modalities agree, the choice of $g$ matters less; when they disagree, $g$ determines which stream dominates.

### 2.3.2 Per-Step Application

For a sequence of length $T$, the fusion is applied **per time step**:

$$
h_{\text{fused}}^{(t)} = g^{(t)} \cdot h_{\text{visual}}^{(t)} + (1 - g^{(t)}) \cdot h_{\text{audio}}^{(t)}, \quad t = 1, \ldots, T
$$

Each step has its own gate $g^{(t)}$, allowing the model to shift attention across time (e.g., favoring audio when the face is briefly occluded).

---

## 2.4 Cross-Modal Conflict Score

### 2.4.1 Cosine Similarity

The **cosine similarity** between the aligned visual and audio vectors is:

$$
\text{sim} = \frac{h_{\text{visual}} \cdot h_{\text{audio}}}{\|h_{\text{visual}}\| \, \|h_{\text{audio}}\|}
$$

where $\cdot$ denotes the dot product and $\|\cdot\|$ the L2 norm. $\text{sim} \in [-1, 1]$; higher values indicate closer alignment in the learned embedding space.

### 2.4.2 Conflict as Cosine Distance

The **Cross-Modal Conflict** score is defined as one minus the similarity, clamped to $[0, 1]$:

$$
C = 1 - \frac{h_{\text{visual}} \cdot h_{\text{audio}}}{\|h_{\text{visual}}\| \, \|h_{\text{audio}}\|} \quad \text{(conceptually)}; \qquad C = \text{clamp}(1 - \text{sim},\; 0,\; 1)
$$

Thus $C \in [0, 1]$. When the modalities **agree** (point in similar directions), $\text{sim}$ is high and $C$ is low. When they **disagree** (point in different or opposite directions), $\text{sim}$ is low and $C$ is high.

### 2.4.3 Interpretation: Cognitive Dissonance and Deception

The conflict score $C$ acts as a **quantitative indicator** of cross-modal incongruence. In affective computing, such incongruence can arise from:

1. **Sarcasm:** The face displays a smile (positive valence) while the voice carries ironic or hostile prosody (negative valence). The visual and audio embeddings point toward different affective regions; $C$ spikes.

2. **Social masking:** The face shows a neutral or positive expression while the voice betrays stress (e.g., high jitter, tremor). The modalities convey different internal states; $C$ is elevated.

3. **Deception:** Deliberate misalignment between displayed and felt emotion. The conflict score does not directly detect deception, but it flags **incongruence**—a necessary (though not sufficient) condition for many deception scenarios.

4. **Sensor noise:** One modality may be corrupted (e.g., motion blur, background noise). The embeddings disagree; $C$ rises. The gate can down-weight the noisy modality, but the conflict score remains a useful signal for the rule engine (e.g., to lower confidence or trigger overrides).

The psychological interpretation: **cognitive dissonance** occurs when an individual holds conflicting beliefs or displays conflicting signals. The conflict score $C$ quantifies the **affective dissonance** between face and voice—the degree to which the two channels tell different stories about the speaker's state.

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why use a learnable gate ($g$) instead of simple feature concatenation?

### Brief

A learnable gate dynamically routes attention between modalities based on their reliability. When one modality is blinded or noisy, the gate can suppress it, avoiding the processing of dead dimensions. Concatenation forces the network to process all dimensions regardless of reliability, wasting capacity and degrading robustness.

### Detailed

**Concatenation:** $[h_{\text{visual}} \oplus h_{\text{audio}}]$ yields a 512-D (after projection) or 1152-D (before projection) vector. The classifier receives this fixed representation. If the camera is blinded, $h_{\text{visual}}$ is zeros or noise. The concatenated vector still contains 256 (or 384) visual dimensions; the network must learn to zero out the corresponding weights. This requires the network to **detect** unreliability and **compensate**—a harder learning problem. Moreover, the same weights are used for all samples; the network cannot adapt per-sample.

**Gated fusion:** The gate $g$ is computed from both modalities. When the visual stream is uninformative (e.g., zeros), the gate input $[h_{\text{visual}}; h_{\text{audio}}]$ reflects that; the gate can learn to output $g \approx 0$, routing the fused representation to the audio stream. The visual contribution is **multiplied by zero**—it is not processed further. The model explicitly suppresses the unreliable modality rather than learning to ignore it through downstream weights. This is more parameter-efficient and robust.

**Ablation:** In principle, a concatenation-based model could learn similar behavior if given sufficient capacity and data. The gate provides an **inductive bias** that makes this behavior easier to learn and more interpretable (the gate value is directly inspectable).

### Comprehensive

The gate implements **conditional computation**: the effective representation depends on the input. Concatenation implements **fixed computation**: the same transformation is applied regardless of input quality. For multimodal systems deployed in variable conditions (lighting, occlusion, noise), conditional computation is preferable. The gate also provides **interpretability**: the XAI Dashboard displays $g$ as "Visual vs. Audio dominance," allowing users to understand why the model made a given prediction. With concatenation, the contribution of each modality is opaque. Finally, the gate enables **controlled ablation** at inference time: the `blind_v` and `blind_a` flags zero out one modality before the projection; the gate naturally adapts. With concatenation, zeroing one modality still leaves the concatenated structure; the gate's convex combination cleanly reduces to a single-modality representation when one stream is zeroed.

---

## Q2: How does the system mathematically handle the dimensional mismatch between DINOv2 and Wav2Vec 2.0 before fusing them?

### Brief

Linear projections map the 384-D visual and 768-D audio vectors to a shared 256-D space. The projections $W_{\text{vis}} \in \mathbb{R}^{256 \times 384}$ and $W_{\text{aud}} \in \mathbb{R}^{256 \times 768}$ are learned; they align the modalities into a common coordinate system where the gate and fusion operate.

### Detailed

**Mismatch:** DINOv2 outputs $x_{\text{vis}} \in \mathbb{R}^{384}$; Wav2Vec 2.0 outputs $x_{\text{aud}} \in \mathbb{R}^{768}$. Direct concatenation yields 1152-D; direct fusion (e.g., $h = x_{\text{vis}} + x_{\text{aud}}$) is undefined due to incompatible dimensions.

**Projection:** Two linear layers are applied:
$$
h_{\text{visual}} = W_{\text{vis}} x_{\text{vis}} + b_{\text{vis}}, \quad h_{\text{audio}} = W_{\text{aud}} x_{\text{aud}} + b_{\text{aud}}
$$
with $h_{\text{visual}}, h_{\text{audio}} \in \mathbb{R}^{256}$. The projections are **learned** during training on CREMA-D. They perform two roles: (1) **dimensionality reduction** (384→256, 768→256); (2) **semantic alignment**—the 256-D space is learned to be a common representation where visual and audio vectors encoding similar affect are close, and the gate can meaningfully interpolate between them.

**Gate input:** After projection, $[h_{\text{visual}} \oplus h_{\text{audio}}] \in \mathbb{R}^{512}$. The gate weights $W_{\text{gate}} \in \mathbb{R}^{1 \times 512}$ operate on this aligned representation.

### Comprehensive

The projection layers implement a form of **modality-specific embedding** followed by **cross-modal alignment**. Each modality is first mapped to a shared space; the mapping is learned so that the resulting vectors are comparable. This is analogous to joint embedding spaces in vision-language models (e.g., CLIP), where images and text are projected to a common space for similarity computation. The choice of 256-D is a hyperparameter; it could be 128 or 512. The constraint is that both modalities must map to the same dimension for the gate and fusion to be defined. The projections add $384 \times 256 + 768 \times 256 \approx 295K$ parameters—a modest increase that enables the entire gated fusion mechanism.

---

## Q3: What is the physical/psychological interpretation of the Cross-Modal Conflict score spiking during a deception event (like Sarcasm)?

### Brief

When the face and voice convey different affective signals, the visual and audio embeddings point in different directions in the learned space. The conflict score $C = 1 - \text{cos\_sim}$ quantifies this angular separation. During sarcasm, the face may show a smile (positive) while the voice carries irony (negative); $C$ spikes because the modalities disagree. Psychologically, this reflects the **incongruence** between displayed and underlying affect—a hallmark of sarcasm and related phenomena.

### Detailed

**Physical interpretation:** The conflict score is the **cosine distance** between the projected visual and audio vectors. In the 256-D embedding space, each vector encodes the affective content of its modality. When the face says "Happy" and the voice says "Negative," the vectors point toward different regions. Cosine similarity measures the angle between them; $C = 1 - \text{sim}$ is high when the angle is large. The spike is thus a **geometric** manifestation of cross-modal disagreement.

**Sarcasm:** Sarcasm involves the intentional display of a positive expression (e.g., smile) with negative communicative intent (criticism, irony). The face is controlled to show the positive cue; the voice often betrays the negative intent through prosody (flat intonation, emphasis, tone). The visual embedding captures the smile; the audio embedding captures the hostile prosody. They point in different directions; $C$ spikes.

**Psychological interpretation:** From a **display rules** perspective (Ekman & Friesen), individuals often regulate their expressions to conform to social norms. Sarcasm is a case where the displayed expression (smile) and the felt or intended affect (negative) diverge. The conflict score quantifies this **divergence** at the level of the learned embeddings. It does not directly measure "deception" or "sarcasm"—it measures **incongruence**. Sarcasm is one cause of incongruence; others include politeness, masking, or sensor error. The rule engine uses $C$ (and related signals like synchrony incongruence) as a **cue** to trigger overrides (e.g., "Happy" with high conflict → possible sarcasm).

### Comprehensive

The conflict score bridges **computational** and **psychological** constructs. Computationally, it is a differentiable function of the embeddings, used for training and inference. Psychologically, it corresponds to the notion of **affective incongruence**—the degree to which the face and voice tell different stories. In deception research, incongruence is a key indicator: liars may display controlled facial expressions while their voice leaks stress or discomfort. The conflict score does not identify deception per se (that would require ground truth and is context-dependent), but it **flags** situations where such analysis may be relevant. For RHNS, the score is used pragmatically: when $C$ exceeds a threshold (e.g., 0.6), the system may (1) lower confidence in the neural prediction, (2) trigger rule-based overrides (e.g., sarcasm, fake smile), or (3) display a "Conflict Spike" warning in the XAI Dashboard. The interpretation is left to the user or downstream application; the system provides the quantitative signal.
