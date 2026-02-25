# 09 — Symbolic Reasoning Engine

## Chapter Overview

This document details the shift from **pure connectionism** (deep learning) to a **neuro-symbolic** architecture in RHNS v2.0, focusing on the rule-based logic in `utils/fusion.py`. It comprises three sections: (1) theoretical background on neuro-symbolic AI and behavioral psychology; (2) practical implementation of the master overrides; and (3) an FAQ addressing anticipated defense queries.

---

# Part I — Theoretical Background (Neuro-Symbolic AI & Behavioral Psychology)

## 1.1 The Limits of Pure Deep Learning

### 1.1.1 Zero-Shot Composite States

**Pure neural networks** learn mappings from input to output from labeled data. For a model trained on CREMA-D's six base emotions (Angry, Disgust, Fear, Happy, Neutral, Sad), the output space is fixed. To recognize **composite** or **nuanced** states—such as Boredom, Contempt, Panic, or Frustration—the network would require:

1. **Labeled examples** of those states in the training set.
2. **Sufficient quantity** to learn the decision boundaries.
3. **Consistent labeling** across annotators.

CREMA-D does not provide labels for Boredom, Contempt, Panic, or Frustration. These states are **compositions** of base emotions and contextual cues: e.g., Frustration = Angry/Sad + forward lean + fidgeting; Panic = Fear + raised shoulders + self-touching. A pure neural network cannot recognize them in **zero-shot**—without explicit training examples—unless it has learned to compose such concepts from base emotions, which would require either (a) a much larger, more diverse dataset with nuanced labels, or (b) a different training objective that encourages compositional reasoning. Neither is available at scale for affective computing.

### 1.1.2 The Black Box Problem

Deep learning models are **opaque**: given an input, they produce an output, but the internal reasoning is not interpretable. A neural network that predicts "Happy" does not explain *why*—which features (face, voice, posture) contributed, and in what combination. For applications requiring **accountability** (e.g., mental health screening, HR analytics, clinical decision support), this opacity is a liability. Users and regulators demand **explainability**: the ability to audit the logic that led to a given conclusion.

### 1.1.3 Data Efficiency

Training a neural network to recognize nuanced states directly would require synthetic labels (e.g., rule-based assignment) or a new dataset. Synthetic labels introduce **noise** and **circularity**—if rules define the labels, the network learns to mimic the rules, and the rules could have been applied directly. A neuro-symbolic approach avoids this: the neural model learns base emotions from clean CREMA-D labels; the rules encode domain knowledge without requiring labeled examples of nuanced states.

---

## 1.2 The Neuro-Symbolic Paradigm

### 1.2.1 Definition

**Neuro-Symbolic AI** combines **neural networks** (subsymbolic, data-driven) with **symbolic reasoning** (logic, rules, knowledge representation). The neural component handles perception—mapping raw signals to abstract representations. The symbolic component handles reasoning—applying logical rules to those representations to derive higher-level conclusions.

### 1.2.2 RHNS v2.0 Architecture

RHNS v2.0 implements a clear division:

1. **Perception (Neural):** The LSTM + GMF pipeline processes visual and audio latents and produces a **base 6-class prediction** (Angry, Disgust, Fear, Happy, Neutral, Sad). This is a **subsymbolic** process: the model learns from data and does not expose interpretable intermediate logic.

2. **Reasoning (Symbolic):** The `classify_nuanced_state()` function in `utils/fusion.py` receives the neural prediction plus **physical telemetry** (FAU values, posture, gestures, prosody). It applies **boolean logic**—explicit if-then rules—to deduce complex states. When a rule fires, the neural prediction is **overridden**. This is a **symbolic** process: the logic is transparent and auditable.

The hybrid design leverages the strengths of both: neural perception for robust base-emotion recognition from high-dimensional data; symbolic reasoning for compositional states that lack labeled training data.

---

## 1.3 Facial Action Coding System (FACS)

### 1.3.1 Overview

**FACS (Facial Action Coding System)** is a comprehensive, anatomically based system for describing facial movements (Ekman & Friesen, 1978). It decomposes the face into **Action Units (AUs)**—minimal, observable muscle movements. Each AU corresponds to a specific muscle or muscle group. FACS enables precise, objective coding of facial expressions.

### 1.3.2 Key Action Units in RHNS

| AU | Muscle | Description | Affective Relevance |
|----|--------|-------------|---------------------|
| **FAU12** | Zygomaticus Major | Lip corner puller (smile) | Positive affect; social smile |
| **FAU6** | Orbicularis Oculi (pars orbitalis) | Cheek raiser (Duchenne marker) | Genuine vs. polite smile |
| **FAU4** | Corrugator Supercilii | Brow lowerer | Anger, concentration, frustration |

### 1.3.3 Duchenne vs. Masked Smile

A **genuine (Duchenne) smile** involves both FAU12 (lip corners pulled up) and **FAU6** (cheeks raised, crow's feet around eyes). A **polite or masked smile** involves only FAU12—the mouth smiles, but the eyes do not. Combining FAU12 and FAU6 allows the system to distinguish:

- **High FAU12 + High FAU6:** Genuine happiness.
- **High FAU12 + Low FAU6:** Polite, social, or fake smile—possible sarcasm or compliance.
- **Low FAU12 + High FAU6:** Unusual; may indicate discomfort or forced expression.

The rule engine uses FAU12 and FAU6 (along with FAU4) to refine predictions—e.g., when synchrony is incongruent and FAU intensity is high, the system may interpret a smile as sarcastic rather than genuine.

---

# Part II — Practical Implementation (The Master Overrides)

## 2.1 The classify_nuanced_state Logic

### 2.1.1 Interception Point

The `classify_nuanced_state()` function in `utils/fusion.py` is invoked by the InferenceThread **after** the neural classifier produces a base prediction. It receives:

- **neural_prediction:** The 6-class label from `NuancedStateClassifier`.
- **neural_confidence:** The model's confidence [0, 1].
- **fau_data:** Dict with FAU12, FAU6, FAU4 (float [0, 1]).
- **body_data:** Posture and gesture flags (see below).
- **audio_data:** Jitter and intensity.
- **synchrony_incongruent:** Boolean—true when visual peak lagged audio peak by >300 ms.

### 2.1.2 Evaluation Flow

The function implements a **cascade**: rules are evaluated in a fixed order. The **first matching rule** determines the output; if no rule matches, the neural prediction is returned. The order matters—more specific rules (e.g., Panic) are evaluated before more general ones to avoid incorrect overrides.

---

## 2.2 Documenting Specific Rules

### 2.2.1 Temporal Synchrony (Sarcasm / Fake Smile)

**Condition:** `synchrony_incongruent` (visual peak followed audio peak by >300 ms).

**Logic:**
$$
\text{if } \text{synchrony\_incongruent} \land (\text{fau\_intensity} > 0.35) \rightarrow \text{"Happy"} \quad \text{(sarcastic smile)}
$$
$$
\text{if } \text{synchrony\_incongruent} \land (\text{fau\_intensity} \leq 0.35) \rightarrow \text{"Neutral"} \quad \text{(fake smile)}
$$

**Rationale:** Delayed visual peak suggests the expression was staged or reactive rather than spontaneous—a cue for sarcasm or politeness.

### 2.2.2 Boredom-Like

**Condition:** Self-touching (hand-to-face) AND low facial intensity.

**Logic:**
$$
\text{Self\_Touching\_Hands} \land (\max(\text{FAU12}, \text{FAU6}, \text{FAU4}) < 0.3) \rightarrow \text{"Neutral"}
$$

**Rationale:** Hand-to-face with low expression suggests disengagement or boredom.

### 2.2.3 Hiding Stress (Masked Stress)

**Condition:** Neural predicts Neutral, but physical cues indicate stress.

**Logic:**
$$
\text{Neural} = \text{"Neutral"} \land (\text{Shoulder\_Rigidity} > 0.85) \land (\text{Audio\_Jitter} > 0.6) \rightarrow \text{"Fear"}
$$

**Rationale:** A neutral face with rigid posture and high vocal jitter suggests suppressed stress or anxiety—the voice leaks what the face masks.

### 2.2.4 Awkwardness-Like

**Condition:** Self-touching AND smile (FAU12 > 0.4).

**Logic:**
$$
\text{Self\_Touching\_Hands} \land (\text{FAU12} > 0.4) \rightarrow \text{"Neutral"}
$$

**Rationale:** Smile with self-touching (e.g., hand on chin) may indicate discomfort or awkwardness.

### 2.2.5 Controlled Annoyance

**Condition:** Finger tapping AND neural predicts Neutral.

**Logic:**
$$
\text{Finger\_Tapping} \land (\text{Neural} = \text{"Neutral"}) \rightarrow \text{"Angry"}
$$

**Rationale:** Fidgeting with a neutral face may indicate suppressed annoyance.

### 2.2.6 Contempt (Posture-Override)

**Condition:** Neural predicts Neutral or Happy, with postural asymmetry or head tilt.

**Logic:**
$$
(\text{Neural} \in \{\text{"Neutral"}, \text{"Happy"}\}) \land (\text{posture\_asymmetry} \lor \text{head\_tilt} > 10°) \rightarrow \text{"Contempt"}
$$

**Rationale:** Asymmetric posture or head tilt with a neutral/happy face can signal disdain or contempt (e.g., unilateral lip raise, tilted head).

### 2.2.7 Panic (Posture-Override)

**Condition:** Neural predicts Fear, with raised shoulders and self-touching.

**Logic:**
$$
(\text{Neural} = \text{"Fear"}) \land \text{shoulders\_raised} \land \text{Self\_Touching\_Hands} \rightarrow \text{"Panic"}
$$

**Rationale:** Fear combined with defensive posture (raised shoulders) and self-soothing (hand-to-face) suggests panic or acute anxiety.

### 2.2.8 Frustration (Posture-Override)

**Condition:** Neural predicts Angry or Sad, with forward lean and fidgeting.

**Logic:**
$$
(\text{Neural} \in \{\text{"Angry"}, \text{"Sad"}\}) \land (\text{lean} = \text{"forward"}) \land (\text{Finger\_Tapping} \lor \text{Self\_Touching\_Hands}) \rightarrow \text{"Frustration"}
$$

**Rationale:** Forward lean (approach motivation) with fidgeting refines Angry/Sad to Frustration—directed, agitated distress.

---

## 2.3 Explainability via Dual-Output Design

### 2.3.1 XAI Dashboard (Visual Output)

The XAI Dashboard displays the **final state**, **confidence**, and **logic source** (Neural, Override, Posture-Override) in the top header. When a rule fires, the state is shown in yellow (override color) and the logic source indicates the override type. The telemetry panels (FAU bars, posture, gestures) provide the raw inputs that triggered the rule.

### 2.3.2 Auditable Trail (logic_source)

The system satisfies **Explainable AI (XAI)** requirements by providing an **auditable trail of logic**. The `logic_source` field in the return value and `state_container` indicates:

- **"Neural":** The output came from the LSTM classifier; no rule fired.
- **"Override":** A rule fired (e.g., Hiding Stress, Boredom, Sarcasm).
- **"Posture-Override":** A posture/gesture-based rule fired (e.g., Contempt, Panic, Frustration).

This field can be **logged to the terminal** for debugging and transparency. When a rule overrides the neural prediction, the system can print the reasoning—e.g., `"Override: Hiding Stress (Neural=Neutral, rigidity=0.88, jitter=0.72)"`—providing an explicit record of *why* the final state was chosen. The architecture supports this by exposing `logic_source` and all telemetry; the logging layer can be implemented to systematically print overrides when they occur. This dual-output design—visual display plus auditable trail—elevates the system from a black-box classifier to an XAI diagnostic tool.

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why hardcode rules instead of training a neural network to recognize "Boredom" or "Panic" directly?

### Brief

Training a neural network to recognize nuanced states requires labeled data that CREMA-D does not provide. Synthetic labels (from rules) introduce noise and circularity. Hardcoded rules encode domain knowledge (e.g., "Fear + raised shoulders + self-touching → Panic") without requiring labeled examples, and they are interpretable and auditable.

### Detailed

**Data requirement:** A neural network needs thousands of labeled examples per class to learn robust decision boundaries. CREMA-D has six base emotions; it has no labels for Boredom, Panic, Contempt, or Frustration. Collecting such labels at scale is expensive and requires expert annotators. Synthetic labels—e.g., assigning "Panic" when Fear + shoulders_raised + touching—would create a circular dependency: the network would learn to mimic the rules, and the rules could have been applied directly without training.

**Domain knowledge:** Nuanced states are **compositionally defined** in the psychological and paralinguistic literature. Panic = Fear + defensive posture + self-soothing. Frustration = Angry/Sad + approach posture + fidgeting. These definitions are well-established; encoding them as rules leverages prior knowledge without requiring the network to rediscover them from data.

**Interpretability:** Rules are transparent. When the system outputs "Panic," we can trace the logic: Neural=Fear, shoulders_raised=True, Self_Touching_Hands=True. A neural network trained to output "Panic" directly would not provide this trace. For diagnostic and accountability applications, interpretability is a requirement.

### Comprehensive

The choice reflects **sample efficiency** and **epistemological** considerations. **Sample efficiency:** Rules require zero labeled examples of nuanced states. The neural model is trained only on base emotions (abundant in CREMA-D). The rules then compose nuanced states from base predictions + physical cues. This is far more data-efficient than training an end-to-end model on synthetic nuanced-state labels. **Epistemological:** Nuanced states are *defined* by combinations of base emotions and contextual cues. The rules implement these definitions explicitly. A neural network would need to *learn* the definitions from data—but the data (synthetic labels) would be generated by the same definitions. The rules avoid this circularity. **Maintainability:** Rules can be updated based on deployment feedback or new psychological findings without retraining the neural model. A neural model would require retraining and potentially new data. The hybrid design maximizes the use of reliable data (CREMA-D base emotions) and reliable knowledge (psychological rules) while minimizing reliance on unreliable synthetic labels.

---

## Q2: How does the system resolve conflicts between a high-confidence neural prediction and a triggering symbolic rule?

### Brief

When a symbolic rule fires, it **overrides** the neural prediction regardless of confidence. The rule takes precedence. The override confidence is set by the rule (e.g., 0.85 or 0.90) rather than the neural confidence. The design prioritizes physical evidence (posture, gestures, prosody) over the neural output when specific conditions are met.

### Detailed

**Override semantics:** The rule engine uses a **cascade**: the first matching rule determines the output. If a rule matches, the function returns immediately with the rule's state and confidence. The neural prediction and its confidence are **discarded** for that sample. There is no weighted combination or confidence threshold—the rule fires when its boolean condition is true.

**Rationale:** The rules are designed to capture cases where the **physical telemetry** contradicts or refines the neural prediction. For example, the neural model may output "Neutral" with 0.9 confidence (the face appears neutral), but high jitter and rigid shoulders indicate stress. The rule overrides to "Fear" because the physical evidence is considered more reliable for detecting masked stress. The neural model was trained on congruent data; it may not have learned to detect such incongruence. The rule encodes domain knowledge that the neural model lacks.

**Confidence assignment:** Override rules assign fixed confidences (e.g., 0.85, 0.88, 0.90) rather than passing through the neural confidence. This reflects the rule's certainty given that its conditions were met. The rule engine does not model uncertainty in the rule itself; it assumes that when the conditions hold, the override is correct.

### Comprehensive

The conflict resolution strategy is **rule-dominant**: when a rule fires, it wins. This is a design choice that prioritizes **interpretability** and **domain knowledge** over the neural model's confidence. Alternative designs could (a) require the rule condition to hold *and* neural confidence to be below a threshold, or (b) blend the neural and rule outputs by confidence. The current design assumes that the rules are **conservative**—they fire only when the physical evidence strongly supports the override. If a rule fires, the system trusts it. If the rules are too aggressive in practice, they can be refined (e.g., adding stricter thresholds) without changing the neural model. The `use_rules` flag allows users to disable the rule engine entirely for ablation, in which case the neural prediction is always returned.

---

## Q3: How does the terminal logging feature elevate the system from a standard classifier to an Explainable AI (XAI) diagnostic tool?

### Brief

A standard classifier outputs only a label and (optionally) confidence. An XAI diagnostic tool provides an **auditable trail**—a record of the reasoning that led to the output. The `logic_source` field and the ability to log rule triggers to the terminal provide this trail. Users and auditors can see *why* the system said "Panic" (e.g., "Override: Panic (Neural=Fear, shoulders_raised=True, touching=True)").

### Detailed

**Standard classifier:** Outputs a label (e.g., "Panic") and perhaps a confidence score. There is no explanation. If the prediction is wrong or controversial, there is no way to debug or audit the logic.

**XAI diagnostic tool:** Outputs the label, confidence, **and** the reasoning. The reasoning includes: (1) whether the output came from the neural model or a rule override; (2) which rule fired (inferred from logic_source and the override state); (3) the physical telemetry that triggered the rule. Terminal logging can print this explicitly—e.g., on each override, log the rule name, the neural prediction, and the key telemetry values. This creates a **timestamped audit trail** that can be reviewed offline for quality assurance, debugging, or regulatory compliance.

**Dual output:** The XAI Dashboard provides **real-time visual** feedback (state, telemetry, logic source). The terminal provides **persistent textual** feedback (the reasoning log). The visual output supports the user during interaction; the textual output supports post-hoc analysis and accountability.

### Comprehensive

Explainability has multiple dimensions: **transparency** (the user can see how the system works), **interpretability** (the user can understand individual predictions), and **auditability** (there is a record for review). The neuro-symbolic architecture provides transparency through the explicit rule logic. The logic_source and telemetry provide interpretability for each prediction. Terminal logging provides auditability—a persistent, timestamped record of the reasoning. For a **diagnostic tool** (e.g., in mental health screening or stress assessment), auditors and clinicians need to verify that the system's conclusions are grounded in observable evidence. The rule-based overrides are grounded in FAU values, posture flags, and prosody—all of which are displayed in the telemetry panels and can be logged. A purely neural system would not provide this grounding; its reasoning is opaque. The terminal logging feature, combined with the rule engine and logic_source, elevates RHNS from a classifier to an XAI diagnostic tool that supports accountability and trust.
