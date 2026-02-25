# 01 — Abstract and Problem Statement

## Chapter Overview

This document establishes the theoretical foundation, practical motivation, and architectural rationale for **RHNS v2.0 (Real-Time Nuanced Human State Recognition)**. It comprises three principal sections: (1) a literature review situating the work within affective computing and identifying the cross-modal gap; (2) a description of the proposed neuro-symbolic solution; and (3) an FAQ addressing anticipated defense queries with tiered responses.

---

# Part I — Theoretical Background (Literature Review)

## 1.1 The Six-Class Paradigm: History and Limitations

### 1.1.1 Ekman's Basic Emotions

The dominant framework for computational emotion recognition traces its origins to **Paul Ekman's theory of basic emotions** (Ekman, 1992; Ekman & Friesen, 1971). Ekman proposed a small set of **universal, discrete emotions**—Anger, Disgust, Fear, Happiness, Sadness, and Surprise—arguing that these are biologically hardwired, cross-culturally recognizable, and associated with distinct facial configurations (Facial Action Coding System, FACS). This paradigm has been adopted almost universally in affective computing: datasets such as CK+, FER2013, RAVDESS, and CREMA-D are structured around six (or seven, with Surprise) discrete classes.

**Computational appeal:** The six-class framework is tractable. It yields balanced classification problems, interpretable confusion matrices, and straightforward evaluation metrics. Deep learning models trained on such datasets achieve high accuracy on held-out test sets drawn from the same distribution.

**Limitations:** The framework assumes that (a) emotions are discrete rather than dimensional, (b) facial expressions are reliable indicators of internal state, and (c) modalities (face, voice, body) are congruent—i.e., they convey the same affective signal. Each assumption is contested in contemporary psychology and fails in real-world deployment.

### 1.1.2 Discrete Labels and Affective Gradations

Human affect exists along **continua** (Russell, 1980; Barrett, 2017). Annoyance and rage, concern and terror, contentment and euphoria occupy different points on valence–arousal or similar dimensional spaces. Six-class systems force a hard assignment that discards this information. Moreover, **intensity** and **blended emotions** (e.g., bittersweet, anxious excitement) are collapsed into single labels, limiting the granularity of downstream applications.

### 1.1.3 Congruence Assumption

Standard emotion datasets are constructed under an implicit **congruence assumption**: the face, voice, and (when present) body are aligned in expressing the same emotion. Actors are instructed to portray a single affective state; recordings are labeled with that state. Models trained on such data learn to fuse modalities under the assumption that agreement is the norm. When modalities disagree—as they frequently do in naturalistic settings—these models lack the representational and reasoning capacity to detect or interpret the disagreement.

---

## 1.2 The Cross-Modal Gap in Affective Computing

### 1.2.1 Definition

The **Cross-Modal Gap** refers to the failure of models trained on **purely congruent data** (e.g., standard CREMA-D) to generalize to scenarios where visual and auditory channels convey **contradictory** affective signals. In such scenarios, the "ground truth" is ambiguous: the face may say "Happy" while the voice signals distress, and the correct interpretation depends on context, intent, and social norms.

### 1.2.2 Why Congruent Training Fails

Models trained on congruent data learn to:

1. **Fuse modalities by averaging or weighting** toward a single label. When modalities disagree, the model produces a compromise prediction (e.g., "Neutral") that may be semantically meaningless.
2. **Ignore conflict as noise.** There is no training signal that rewards the model for *detecting* disagreement; the loss function penalizes deviation from a single ground-truth label.
3. **Lack explicit reasoning over contradiction.** End-to-end neural networks do not naturally produce interpretable outputs such as "face suggests Happy, voice suggests Stress—possible sarcasm or masking."

Consequently, such models **fail completely** in real-world scenarios where human states are contradictory—e.g., smiling while experiencing high vocal stress, or maintaining a neutral face while the voice betrays anxiety.

### 1.2.3 Real-World Prevalence

Contradictory states are not edge cases. They arise from:

- **Social masking:** Display rules (Ekman & Friesen, 1975) dictate that individuals suppress or exaggerate expressions in social contexts. A service worker may smile while feeling frustrated; a patient may appear calm while voice tremors indicate fear.
- **Sarcasm and irony:** The face and voice intentionally convey opposing valences for communicative effect.
- **Stress leakage:** Attempts to maintain a composed face may fail in the voice (e.g., increased jitter, pitch variability) due to autonomic arousal.

Affective computing systems that assume congruence will misclassify or miss these states entirely.

---

## 1.3 Nuanced Human States: Psychological and Computational Perspectives

### 1.3.1 Definition

**Nuanced human states** are affective configurations that (a) are not reducible to a single basic emotion, (b) often involve **cross-modal incongruence**, and (c) require inference over the *relationship* between modalities rather than over each modality in isolation. From a psychological perspective, they reflect the complexity of human affect regulation, social display rules, and the distinction between felt and displayed emotion.

### 1.3.2 Sarcasm

**Psychological:** Sarcasm involves the intentional expression of a positive valence (e.g., smile) paired with negative communicative intent (irony, criticism). The face may show a smile (FAU12), but the voice carries flat, hostile, or exaggerated prosody. Listeners resolve the contradiction through pragmatic inference.

**Computational:** Detecting sarcasm requires (1) recognizing the visual signal (e.g., Happy), (2) recognizing the acoustic signal (e.g., negative prosody, flat intonation), and (3) flagging their **incongruence** rather than collapsing to a single label. Standard six-class models perform (1) and (2) but not (3).

### 1.3.3 Social Masking

**Psychological:** Social masking refers to the deliberate suppression or alteration of emotional expression to conform to social norms. The displayed face may be neutral or positive while the internal state is negative. Masking is common in professional, clinical, and caregiving contexts.

**Computational:** Masking is detectable when the face suggests one state (e.g., Neutral) while **prosodic leakage** (vocal jitter, intensity, tremor) suggests another (e.g., stress, anxiety). Models must attend to the *discrepancy* and infer the masked state.

### 1.3.4 Hiding Stress (Masked Stress)

**Psychological:** Hiding stress is a specific form of masking where the individual attempts to appear calm or neutral while experiencing internal arousal. The face may be controlled, but the voice—which is harder to voluntarily regulate—betrays stress through increased pitch perturbation (jitter), tremor, or intensity fluctuations.

**Computational:** Detection requires (1) a neutral or low-arousal facial prediction, (2) high acoustic stress indicators (e.g., jitter > 0.6, rigidity in posture), and (3) a rule or learned mechanism that overrides the face-dominant prediction to "Fear" or "Stress" when the discrepancy is detected.

---

# Part II — Practical Learning & Proposed Solution (The RHNS v2.0 Approach)

## 2.1 Addressing the Limitations

The RHNS v2.0 architecture is designed to address each limitation outlined in Part I:

| Limitation | RHNS v2.0 Response |
|------------|---------------------|
| Discrete six-class collapse | Neural classifier outputs base emotions; symbolic engine refines to nuanced states (Contempt, Panic, Frustration) when geometric/prosodic rules fire |
| Congruence assumption | Gated Multimodal Fusion + Conflict Attention explicitly model cross-modal disagreement; conflict score $\in [0,1]$ quantifies incongruence |
| No reasoning over contradiction | Symbolic Reasoning Engine applies overrides when conflict, posture, or prosody indicate sarcasm, masking, or hiding stress |
| Lack of interpretability | XAI Dashboard and logic_source ("Neural" vs "Override") provide transparency; terminal logging supports defense and debugging |

## 2.2 The Neuro-Symbolic Hybrid

### 2.2.1 Architectural Division of Labor

RHNS v2.0 implements a **clear separation of concerns**:

1. **Deep Learning (Perception):** The neural pipeline—DINOv2, Wav2Vec 2.0, LSTM, GMF—serves as a **perception engine**. It extracts high-dimensional representations from face and voice, fuses them with a learned gate, and produces a **base six-class prediction** (Angry, Disgust, Fear, Happy, Neutral, Sad) plus auxiliary outputs (gate weight, conflict score). This stage is trained on CREMA-D and learns to recognize **congruent** emotional expressions.

2. **Symbolic Reasoning (Nuanced States):** The rule-based engine (`utils/fusion.py`) operates **after** the neural classifier. It receives the base prediction, FAU values, posture metrics, prosody (jitter, intensity), and conflict flag. It applies **geometric and prosodic rules** to override the neural output when specific conditions indicate nuanced states—e.g., Neutral + high jitter + rigid shoulders → "Fear" (Hiding Stress); Fear + shoulders raised + hand-to-face → "Panic"; Happy/Neutral + posture asymmetry → "Contempt".

### 2.2.2 Transition from Pure Deep Learning

A **pure end-to-end neural network** would require (a) training data with nuanced state labels (Sarcasm, Hiding Stress, etc.), which CREMA-D does not provide, and (b) sufficient examples of cross-modal contradiction for the network to learn the mapping. Such data are scarce and expensive to collect.

The neuro-symbolic approach **decouples** perception from reasoning:

- **Perception** is learned from abundant congruent data (CREMA-D).
- **Reasoning** is encoded as explicit rules derived from psychological and paralinguistic literature. Rules can be updated without retraining the neural model.

This design allows the system to recognize nuanced states **without** native labels for those states in the training set. The neural model provides base emotions and conflict; the symbolic engine interprets their combination.

### 2.2.3 Conflict as a First-Class Signal

The GMF layer computes a **conflict score**:

$$
\text{conflict} = \text{clamp}(1 - \text{cosine\_sim}(h_{\text{visual}}, h_{\text{audio}}),\; 0,\; 1)
$$

This score is not merely an internal representation—it is **exposed** to the symbolic engine (via `synchrony_incongruent` and related logic) and to the user (via the XAI Dashboard). When conflict exceeds a threshold (e.g., 0.6), the system can trigger overrides (e.g., sarcasm, fake smile) or flag the prediction for human review. Conflict is thus a **first-class signal** for reasoning, not a side effect of fusion.

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why use the CREMA-D dataset if it does not contain natively labeled nuanced states like "Sarcasm"?

### Brief

CREMA-D provides high-quality, congruent multimodal (face + voice) data for the six base emotions. RHNS uses it to train the **perception** stage. Nuanced states are inferred by the **symbolic engine** from the combination of base predictions, conflict, and geometric/prosodic features—not from CREMA-D labels.

### Detailed

CREMA-D offers 7,442 clips with consistent labeling, 30 FPS video, and 16 kHz audio—ideal for training a multimodal base-emotion classifier. No widely available dataset provides native labels for Sarcasm, Hiding Stress, or Contempt at scale. Rather than collecting costly new data, RHNS adopts a **transfer-and-reason** strategy: train perception on CREMA-D, then apply domain knowledge (rules) to reinterpret outputs when conflict or posture indicates nuanced states. The symbolic rules are informed by psychological literature (e.g., vocal jitter as stress indicator, posture asymmetry as contempt cue) and do not require labeled examples of those states.

### Comprehensive

The choice reflects a **data efficiency** and **generalization** argument. (1) **Data efficiency:** Labeled nuanced-state data are rare; CREMA-D is abundant. Training only the base-emotion classifier on CREMA-D avoids the need for a large nuanced-state corpus. (2) **Compositional reasoning:** Nuanced states are *compositions* of base emotions and contextual cues. Sarcasm = Happy (face) + negative prosody + temporal incongruence. Hiding Stress = Neutral (face) + high jitter + rigid posture. The symbolic engine implements this composition explicitly. (3) **Updatability:** Rules can be refined based on deployment feedback without retraining the neural model. (4) **Interpretability:** The system can explain *why* it overrode to "Hiding Stress" (e.g., "Neural: Neutral; Override: jitter=0.72, rigidity=0.88"). An end-to-end model would not naturally produce such explanations.

---

## Q2: What is the primary advantage of a Neuro-Symbolic architecture over an End-to-End Neural Network for this specific task?

### Brief

Neuro-symbolic systems separate **perception** (learned from data) from **reasoning** (encoded as rules). For nuanced states that lack large labeled datasets, rules can capture domain knowledge (e.g., "high jitter + neutral face → possible stress masking") without requiring thousands of labeled examples. The architecture is also more interpretable and easier to update.

### Detailed

An end-to-end neural network would need to learn the mapping from raw inputs to nuanced states (Sarcasm, Hiding Stress, Contempt, etc.) from data. Such data are scarce. The network would also need to learn *when* to trust face vs. voice, and *when* to flag contradiction—all from examples. In contrast, the neuro-symbolic design: (1) uses the neural model only for what it does well—recognizing base emotions from congruent data; (2) uses rules to encode well-established relationships (jitter ↔ stress, posture asymmetry ↔ contempt) that do not require massive datasets; (3) produces interpretable overrides ("Override: Hiding Stress because jitter=0.7, rigidity=0.9"); (4) allows rule updates without retraining. The primary advantage is **sample efficiency** and **interpretability** for states that are compositionally defined but poorly represented in existing datasets.

### Comprehensive

The advantage rests on three pillars. **Epistemological:** Nuanced states are defined by *relationships* (incongruence, leakage, display rules) that psychologists have characterized. Encoding these as rules leverages prior knowledge; learning them purely from data would require orders of magnitude more examples. **Architectural:** The neural model outputs a probability distribution over six classes plus a conflict score. The symbolic engine operates on these *abstract* outputs, not raw pixels or waveforms. This abstraction reduces the complexity of the reasoning problem and makes it amenable to rule-based logic. **Operational:** In deployment, users and clinicians need to understand *why* the system said "Hiding Stress." A rule-based override ("Neutral face + jitter 0.72 + shoulder rigidity 0.88") is auditable; a black-box neural output is not. For applications requiring accountability (e.g., mental health screening, HR analytics), interpretability is a requirement, not a convenience.

---

## Q3: How does the system define "Modality Conflict" mathematically versus conceptually?

### Brief

**Conceptually:** Conflict occurs when the face and voice convey different affective signals (e.g., smile + strained voice). **Mathematically:** Conflict is $1 - \text{cosine\_similarity}(h_{\text{visual}}, h_{\text{audio}})$, clamped to $[0, 1]$, where higher values indicate greater disagreement in the learned embedding space.

### Detailed

**Conceptual definition:** Modality conflict is the **incongruence** between affective signals across channels. A person may smile (positive valence) while speaking with flat or hostile prosody (negative valence). Conceptually, conflict is a *semantic* notion: the modalities "disagree" about the affective state. The system also uses a **temporal** notion: when the visual peak (e.g., smile intensity) follows the audio peak by more than 300 ms, the expression may be delayed or staged (sarcasm, fake smile).

**Mathematical definition:** The GMF layer projects visual and audio latents into a shared 256-D space: $h_{\text{visual}} = W_{\text{vis}} x_{\text{vis}}$, $h_{\text{audio}} = W_{\text{aud}} x_{\text{aud}}$. Conflict is:

$$
\text{conflict} = \text{clamp}\left(1 - \frac{h_{\text{visual}} \cdot h_{\text{audio}}}{\|h_{\text{visual}}\| \|h_{\text{audio}}\|},\; 0,\; 1\right)
$$

Cosine similarity $\in [-1, 1]$; when modalities agree, similarity is high and conflict is low; when they disagree, similarity is low and conflict is high. The projection matrices $W_{\text{vis}}$, $W_{\text{aud}}$ are learned, so the embedding space is task-specific—vectors that are "far" in this space are those the model has learned to treat as affectively discordant.

### Comprehensive

The mathematical and conceptual definitions are **aligned but not identical**. **Conceptually,** conflict is defined over *meaning* (face says X, voice says Y). **Mathematically,** conflict is defined over *embedding distance* in a learned space. The alignment is achieved through training: the model is trained on CREMA-D, where congruent samples have matching labels. The projection layers learn to map congruent face–voice pairs to similar $h_{\text{visual}}$, $h_{\text{audio}}$, and incongruent pairs (which appear as "noise" or minority cases in the data) to dissimilar vectors. Thus, the learned conflict score is a **proxy** for semantic incongruence. It is not a perfect proxy—the model has never seen explicitly labeled "sarcasm" or "masking" during training—but the symbolic engine uses it as a *cue* to trigger overrides when conflict exceeds a threshold. The temporal notion of conflict (synchrony incongruence: visual peak lags audio by >300 ms) is a separate, rule-based signal that complements the learned cosine-based score. Together, they provide both a continuous measure of embedding-space disagreement and a discrete flag for temporal misalignment.

---

## References (Representative)

- Barrett, L. F. (2017). *How Emotions Are Made: The Secret Life of the Brain*. Houghton Mifflin Harcourt.
- Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion*, 6(3-4), 169–200.
- Ekman, P., & Friesen, W. V. (1971). Constants across cultures in the face and emotion. *Journal of Personality and Social Psychology*, 17(2), 124–129.
- Ekman, P., & Friesen, W. V. (1975). *Unmasking the Face*. Prentice-Hall.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
