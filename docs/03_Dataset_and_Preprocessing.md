# 03 — Dataset and Preprocessing

## Chapter Overview

This document details the CREMA-D dataset, the mathematical challenges of class imbalance in multimodal affective recognition, and the specific **"Double-Dipping"** collapse that the RHNS v2.0 project overcame during development. It comprises three sections: (1) theoretical background on CREMA-D and class imbalance; (2) practical learning and the Double-Dipping fix; and (3) an FAQ addressing anticipated defense queries.

---

# Part I — Theoretical Background (CREMA-D & Class Imbalance)

## 1.1 The CREMA-D Dataset

### 1.1.1 Overview

**CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)** is a widely adopted benchmark for multimodal affective computing. It comprises **7,442 original clips** from **91 actors** (48 male, 43 female, ages 20–74) representing diverse ethnic backgrounds. Each actor performed **12 sentences** expressed in **six base emotions** at four intensity levels (Low, Medium, High, Unspecified).

### 1.1.2 The Six Base Emotions

| Emotion | Code | Description |
|---------|------|-------------|
| Happy | HAP | Positive valence, high arousal |
| Sad | SAD | Negative valence, low arousal |
| Angry | ANG | Negative valence, high arousal |
| Fear | FEA | Negative valence, high arousal |
| Disgust | DIS | Negative valence, variable arousal |
| Neutral | NEU | Low valence, low arousal |

### 1.1.3 Labeling and Quality

Emotion labels were obtained via **crowd-sourcing**: 2,443 participants each rated 90 unique clips (30 audio-only, 30 visual-only, 30 audiovisual). Over 95% of clips have more than seven ratings, ensuring high inter-rater agreement. The resulting labels are thus **consensus-based** rather than actor-intended, which improves reliability for supervised learning.

### 1.1.4 Why CREMA-D Is the Academic Standard

CREMA-D is preferred over alternatives (RAVDESS, IEMOCAP, etc.) for multimodal affective research because:

1. **Synchronized audio-visual streams:** Video and audio are temporally aligned and recorded under controlled conditions. There is no lip-sync drift or modality mismatch.

2. **High-quality recordings:** Professional capture ensures consistent lighting, resolution (30 FPS video), and audio (16 kHz WAV). The data are suitable for deep learning without extensive preprocessing.

3. **Scale and diversity:** 7,442 clips provide sufficient data for training modern architectures; 91 actors and multiple sentences reduce actor-specific overfitting.

4. **Standardized format:** Filename convention `{actor_id}_{modality}_{emotion}_{intensity}` (e.g., `0032_DFA_ANG_XX`) enables programmatic label extraction without external metadata files.

5. **Multimodal design:** The dataset was explicitly designed for audiovisual fusion, with matched audio and video for each clip—unlike datasets that combine modalities from separate sources.

---

## 1.2 The Class Imbalance Problem

### 1.2.1 Imbalance in Base Emotions

CREMA-D exhibits **significant class imbalance** at the base-emotion level. Certain emotions (e.g., Happy, Neutral) appear far more frequently than others (e.g., Fear, Disgust) due to actor availability, sentence selection, and crowd-sourcing participation. The long-tail distribution means that a model trained with standard cross-entropy and uniform sampling will be biased toward majority classes, underperforming on minority classes.

### 1.2.2 The Nuanced-State Imbalance Challenge

A more severe challenge arises when mapping balanced base emotions to a highly imbalanced set of **Nuanced States**. If one were to define nuanced states (Sarcasm, Hiding Stress, Awkwardness, Contempt, Panic, Frustration, etc.) and assign synthetic labels—e.g., via heuristics or rules—the resulting distribution could be **extremely skewed**. For instance, "Mixed Feelings" or "Neutral-with-Leakage" might comprise 70%+ of the data, while "Sarcasm" or "Panic" might appear in <2% of samples. Training a neural network on such a distribution without careful handling leads to:

- **Majority-class bias:** The model predicts the dominant class for most inputs.
- **Minority-class neglect:** Rare classes receive insufficient gradient signal.
- **Evaluation distortion:** Accuracy can be misleading when the majority class dominates (e.g., 70% accuracy by always predicting "Mixed Feelings").

---

## 1.3 Standard Mitigation Strategies

### 1.3.1 Cost-Sensitive Learning (Inverse Frequency Weighting)

**Idea:** Penalize errors on minority classes more heavily by weighting the loss. For cross-entropy:

$$
\mathcal{L} = -\sum_{i=1}^{N} w_{c(y_i)} \log p(y_i \mid x_i)
$$

where $w_c = N / n_c$ is the inverse frequency weight for class $c$, $N$ is total samples, and $n_c$ is the count of class $c$. Thus $w_c$ is larger for minority classes, so a misclassification of a minority-class sample contributes more to the loss.

**Advantage:** No change to the data pipeline; only the loss function is modified.

**Disadvantage:** When $n_c$ is very small, $w_c$ becomes large. Gradients from minority-class samples can dominate and cause instability—especially when those samples are noisy or ambiguous.

### 1.3.2 Data-Level Resampling

**Idea:** Change the *distribution* of samples seen by the model during training, rather than the loss.

- **Oversampling:** Duplicate minority-class samples so each class appears equally often per epoch.
- **Undersampling:** Discard majority-class samples to balance the dataset.
- **Weighted sampling:** Assign each sample a weight $w_i = 1/n_{c(i)}$; the sampler draws samples with probability proportional to $w_i$. Minority-class samples are drawn more frequently.

**Advantage:** The model receives balanced batches; gradient signal from each class is more uniform.

**Disadvantage:** Oversampling can lead to overfitting on repeated minority samples; undersampling discards data. Weighted sampling with replacement avoids duplication but can increase variance.

---

# Part II — Practical Learning & Implementation (The "Double-Dipping" Collapse)

## 2.1 The Initial Failure State: Double-Dipping

### 2.1.1 The Architectural Mistake

**Double-Dipping** refers to the simultaneous application of *both*:

1. **WeightedRandomSampler:** Which forces the DataLoader to present each class with approximately equal probability per epoch (by assigning per-sample weights $w_i = 1/n_{c(i)}$ and sampling with replacement).

2. **CrossEntropyLoss(weight=class_weights):** Which applies inverse-frequency weights $w_c = N/n_c$ inside the loss function.

The intent was to "double down" on balancing—both at the data level and at the loss level. The result was catastrophic.

### 2.1.2 Why Double-Dipping Is Redundant and Harmful

Under WeightedRandomSampler, each batch is already **approximately class-balanced**. The model sees, on average, the same number of samples per class per epoch. The effective training distribution is uniform over classes. Applying additional inverse-frequency weights in the loss is **redundant**—the data distribution is already balanced—and **harmful**, because it further amplifies the loss contribution of minority-class samples beyond what the balanced sampling already achieves.

---

## 2.2 Symptom Analysis

### 2.2.1 Collapse into Minority-Class Guessing

When both mechanisms are active, the loss landscape becomes distorted. Consider a batch with equal numbers of samples per class. The loss weights $w_c$ make errors on minority classes (e.g., Fear, Disgust) contribute 3–5× more to the total loss than errors on majority classes. The optimizer is incentivized to **reduce loss on minority classes at all costs**. The model learns to predict minority classes (Fear, Disgust) more frequently—even when the true label is a majority class (Happy, Neutral)—because doing so avoids the massive loss spikes from misclassifying minority-class samples.

**Result:** The model effectively **abandons** majority classes. It over-predicts minority classes to "protect" itself from the weighted loss. Predictions collapse toward a subset of minority classes.

### 2.2.2 Confusion Matrix: Vertical Stripes

The failure mode manifests in the **confusion matrix** as **vertical stripes**. A vertical stripe in column $j$ indicates that the model predicts class $j$ for many different true labels. When the model collapses to minority-class guessing, one or two predicted classes (e.g., Disgust, Fear) show high counts across *all* rows (true labels). The matrix exhibits strong vertical concentration: many true Angry, Happy, Neutral, and Sad samples are incorrectly predicted as Disgust or Fear. Early training runs produced precisely this pattern before the Double-Dipping fix was applied.

---

## 2.3 The Mathematical Fix

### 2.3.1 Correction

The fix is to **remove** the inverse-frequency weights from the loss function and rely **solely** on the WeightedRandomSampler for balancing.

**Before (Double-Dipping):**
```python
criterion = nn.CrossEntropyLoss(weight=class_weights)
train_loader = DataLoader(..., sampler=WeightedRandomSampler(...))
```

**After (Corrected):**
```python
criterion = nn.CrossEntropyLoss()  # No weight argument
train_loader = DataLoader(..., sampler=WeightedRandomSampler(...))
```

### 2.3.2 Mathematical Justification

Let $p_c$ denote the probability of sampling a sample from class $c$ under the WeightedRandomSampler. With $w_i = 1/n_c$ for samples in class $c$, we have $p_c \propto n_c \cdot (1/n_c) = 1$ (after normalization). Thus each class is sampled with equal probability. The effective per-sample loss over an epoch is:

$$
\mathbb{E}_{\text{sampler}}[\mathcal{L}] \approx -\frac{1}{K}\sum_{c=1}^{K} \frac{1}{n_c}\sum_{i \in c} \log p(y_i \mid x_i)
$$

With **standard** cross-entropy (no $w_c$ in the loss), the gradient signal from each class is already balanced by the sampling distribution. Adding $w_c$ in the loss would **over-weight** minority classes, since they are already seen equally often. The corrected approach achieves balance through sampling alone, avoiding the gradient instability of extreme loss weights.

---

## 2.4 The Baseline Pivot

### 2.4.1 Strategic Decision

A further architectural decision was the **Baseline Pivot**: train the Deep Learning model strictly on the **6 balanced base emotions** (Angry, Disgust, Fear, Happy, Neutral, Sad), and offload **Nuanced State** logic (Sarcasm, Panic, Contempt, Frustration, Hiding Stress, etc.) to the **Symbolic Rule Engine**.

### 2.4.2 Rationale

1. **Label availability:** CREMA-D provides reliable labels only for the 6 base emotions. Nuanced states would require synthetic labels (e.g., rule-based assignment), introducing **label noise** and ambiguity.

2. **Data efficiency:** Training on 6 classes with CREMA-D yields sufficient samples per class. Training on 10+ nuanced states would require either (a) synthetic labels (noisy) or (b) a different dataset with native nuanced-state annotations (scarce).

3. **Compositional reasoning:** Nuanced states are *compositions* of base emotions and contextual cues. The rule engine implements this explicitly: e.g., Neutral + high jitter + rigid shoulders → Hiding Stress. The neural model need not learn these compositions from data.

4. **Interpretability:** Rule-based overrides are auditable; a neural model predicting "Sarcasm" directly would not explain *why*.

By pivoting to 6 base emotions for the LSTM and reserving nuanced states for the rule engine, the project avoids synthetic label noise and leverages the strengths of both paradigms.

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why was CREMA-D chosen over other multimodal datasets like RAVDESS or IEMOCAP?

### Brief

CREMA-D offers 7,442 synchronized audio-visual clips with crowd-sourced labels, controlled recording conditions, and a standardized format. RAVDESS is smaller and less diverse; IEMOCAP is richer in naturalistic speech but smaller and more expensive to use. CREMA-D provides the best balance of scale, quality, and accessibility for training a multimodal base-emotion classifier.

### Detailed

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):** Contains 7,356 clips from 24 actors, 8 emotions. It is smaller in actor diversity and does not match CREMA-D's clip count when considering the 6 overlapping emotions. RAVDESS includes "surprise" and "calm," which are not in CREMA-D's 6-class scheme; aligning the label spaces requires mapping or dropping classes.

**IEMOCAP (Interactive Emotional Dyadic Motion Capture):** Contains ~10 hours of dyadic conversations from 10 actors. It is naturalistic and includes nuanced annotations (e.g., frustration, excitement), but the scale is smaller (fewer clips), the format is more complex (sessions, turns), and licensing can be restrictive. IEMOCAP is better suited for conversational affect than for training a general-purpose base-emotion classifier.

**CREMA-D** provides: (1) 7,442 clips—sufficient for deep learning; (2) 91 actors—reduced actor bias; (3) synchronized A/V—no alignment preprocessing; (4) 6 emotions—direct match to the target taxonomy; (5) crowd-sourced labels—high agreement; (6) open access—no licensing barriers. For RHNS v2.0's goal of training a base-emotion perception engine, CREMA-D is the most suitable choice.

### Comprehensive

The selection criterion was **fitness for purpose**: training a multimodal LSTM + GMF model to recognize 6 base emotions from temporal sequences of visual and acoustic embeddings. CREMA-D satisfies this because: (1) **Modality alignment:** Each clip has matched video and audio; no cross-dataset stitching. (2) **Temporal structure:** Clips are 1–3 seconds, suitable for 15-frame sequences. (3) **Label consistency:** Crowd-sourcing yields consensus labels; actor-intended labels can be ambiguous. (4) **Reproducibility:** The dataset is publicly available with a clear license; RAVDESS and IEMOCAP have different access terms. (5) **Benchmark comparability:** CREMA-D is widely used in the literature, enabling direct comparison with prior work. The choice reflects a trade-off: CREMA-D lacks naturalistic nuance (e.g., spontaneous sarcasm), but provides reliable, large-scale base-emotion data. The RHNS architecture compensates by using the rule engine for nuanced states, so CREMA-D's limitation is acceptable.

---

## Q2: Explain the mathematical phenomenon of "Double-Dipping" and why it causes a neural network to collapse into minority-class guessing.

### Brief

Double-Dipping applies both weighted sampling (balanced batches) and inverse-frequency loss weights. The loss weights over-penalize errors on minority classes. The optimizer minimizes loss by predicting minority classes more often, even when wrong, to avoid the large penalties from misclassifying minority samples. The model collapses toward minority-class predictions.

### Detailed

**Setup:** Let $n_c$ be the count of class $c$, $N = \sum_c n_c$, and $w_c = N/n_c$. With WeightedRandomSampler, each class is seen with roughly equal frequency. With CrossEntropyLoss(weight=$w$), the loss for a sample $(x_i, y_i)$ is:

$$
\ell_i = -w_{y_i} \log p(y_i \mid x_i)
$$

For a minority class $c$ with small $n_c$, $w_c$ is large. A single misclassification of a minority-class sample contributes $w_c \cdot \log(\text{small}) \approx w_c \cdot (\text{large negative})$, i.e., a massive positive loss. The gradient from that sample is correspondingly large.

**Collapse mechanism:** The optimizer seeks to reduce total loss. The dominant gradient contributions come from minority-class samples (because $w_c$ is large). To reduce loss, the model learns to predict minority classes more often—even for inputs whose true label is a majority class. Predicting "Disgust" for a Happy sample incurs a moderate penalty ($w_{\text{Happy}}$ is small). Predicting "Happy" for a Disgust sample incurs a huge penalty ($w_{\text{Disgust}}$ is large). The asymmetric penalties incentivize over-prediction of minority classes. The model "collapses" into minority-class guessing to avoid the catastrophic loss spikes from minority-class errors.

### Comprehensive

The phenomenon is a form of **gradient imbalance** exacerbated by **redundant balancing**. Under WeightedRandomSampler, the expected number of gradient updates per class per epoch is already equal. Adding $w_c$ in the loss further amplifies the gradient magnitude for minority-class samples. The effective learning rate for minority classes becomes $w_c$ times larger than for majority classes. In practice, $w_c$ can be 3–5× for tail classes. The optimizer effectively treats minority-class samples as 3–5× more "important," leading to overfitting on those samples and underfitting on majority classes. The collapse is not toward the majority class (as in unweighted imbalanced training) but toward the *minority* class—a counterintuitive result that arises from the double application of balancing. The fix—removing $w_c$ from the loss—restores gradient balance: each class contributes equally to the loss in expectation, and the model learns a balanced decision boundary.

---

## Q3: Why did the architecture pivot from training the LSTM on 10 nuanced states to training on 6 base emotions?

### Brief

Training on 10 nuanced states would require labels that CREMA-D does not provide. Synthetic labels (e.g., from rules) introduce noise and ambiguity. The pivot to 6 base emotions leverages CREMA-D's reliable labels and offloads nuanced-state reasoning to the rule engine, which can encode domain knowledge without labeled data.

### Detailed

**Initial consideration:** An early design explored training the LSTM to predict 10+ nuanced states (Sarcasm, Hiding Stress, Contempt, Panic, Frustration, Awkwardness, etc.) directly. This would require either: (a) a dataset with native nuanced-state annotations (none exists at CREMA-D scale), or (b) synthetic labels derived from rules or heuristics.

**Problems with synthetic labels:** (1) **Noise:** Rule-based assignment is imperfect; e.g., "high jitter + neutral face" might be stress or might be vocal variability. (2) **Imbalance:** Nuanced states would be highly skewed (e.g., "Mixed Feelings" 70%, "Sarcasm" 2%), exacerbating the Double-Dipping risk. (3) **Circularity:** If rules define the labels, the neural model would learn to mimic the rules—redundant with the rule engine itself.

**Pivot rationale:** Train the LSTM on 6 base emotions using CREMA-D's gold-standard labels. Use the rule engine to *compose* nuanced states from base predictions + geometric/prosodic cues. The neural model does perception; the rule engine does reasoning. This separation avoids synthetic label noise, avoids extreme imbalance, and preserves interpretability.

### Comprehensive

The pivot reflects a **division of labor** between data-driven and knowledge-driven components. **Perception** (mapping raw signals to base emotions) benefits from large-scale supervised learning; CREMA-D provides that. **Reasoning** (mapping base emotions + context to nuanced states) benefits from explicit rules; psychological and paralinguistic literature provide that. Training the LSTM on nuanced states would conflate the two: the model would need to learn both perception and composition from data, but the data for composition (nuanced-state labels) are unreliable. By training only on base emotions, the LSTM learns a clean, well-defined task. The rule engine then applies domain knowledge—e.g., "Neutral + jitter > 0.6 + rigidity > 0.85 → Hiding Stress"—without requiring the neural model to infer these relationships from noisy synthetic labels. The architecture thus maximizes the use of reliable data (CREMA-D base emotions) and minimizes the use of unreliable data (synthetic nuanced-state labels).
