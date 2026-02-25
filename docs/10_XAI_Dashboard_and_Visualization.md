# 10 — XAI Dashboard and Visualization

## Chapter Overview

This document details the **Explainable AI (XAI)** principles applied to the RHNS v2.0 user interface and logging mechanisms. It comprises three sections: (1) theoretical background on XAI and cognitive load; (2) practical implementation of the segregated architecture; and (3) an FAQ addressing anticipated defense queries.

---

# Part I — Theoretical Background (Explainable AI and Cognitive Load)

## 1.1 The Black Box Problem in Affective Computing

### 1.1.1 Opaque Predictions

**Neural networks** produce predictions without exposing the reasoning that led to them. Given an input (face, voice), a model outputs a label (e.g., "Happy") and perhaps a confidence score. The internal computation—which features contributed, how modalities were weighted, whether rules were applied—remains hidden. This **opacity** is acceptable for some applications (e.g., image classification for search) but unacceptable for **behavioral analysis** and **affective computing** in high-stakes contexts.

### 1.1.2 Why Opaque Predictions Are Unacceptable

In behavioral analysis, clinicians, researchers, and operators must:

1. **Verify** that the system's conclusion is grounded in observable evidence (FAU values, posture, prosody).
2. **Debug** when the prediction seems wrong—understanding *why* the system said "Panic" enables correction of sensors, rules, or model.
3. **Account** for decisions in regulatory or clinical settings—opaque systems cannot be audited.
4. **Trust** the system—users are reluctant to act on predictions they cannot understand.

A system that outputs "Sarcasm" without explaining that it detected a smile (FAU12) with delayed synchrony and high conflict score provides no basis for verification or trust. The prediction is a **black box**.

### 1.1.3 The Necessity of XAI

**Explainable AI (XAI)** addresses the black box by providing **human-readable justification** for AI decisions. In RHNS v2.0, XAI is implemented through: (1) **visual transparency**—telemetry panels showing FAU, posture, gate weight, conflict; (2) **logic source**—indicating whether the output came from the neural model or a rule override; (3) **terminal logging**—routing the detailed reasoning (rules triggered, conflict score, gate weights) to the system terminal for an auditable diagnostic log. These mechanisms satisfy the requirement that the system justify its conclusions in terms of observable, interpretable evidence.

---

## 1.2 Cognitive Segregation in UI Design

### 1.2.1 The Problem of Information Overload

When raw perceptual data (video, audio) and dense analytical telemetry (FAU values, posture flags, gate weights, conflict scores) are **mixed** in a single visual space—e.g., overlaying all telemetry on the video—the human operator faces **information overload**. The video is the primary focus; overlaying dozens of numeric values and labels obscures the subject and fragments attention. The operator must simultaneously (a) observe the subject's behavior, (b) read the telemetry, and (c) interpret the prediction—a high cognitive load that degrades performance.

### 1.2.2 Design Philosophy: Separation of Concerns

The design philosophy of **cognitive segregation** separates:

1. **Raw perceptual data:** The live video and (implicitly) the audio stream. This is the **primary feed**—what the operator needs to see to understand the subject's behavior.

2. **Dense analytical telemetry:** FAU progress bars, posture matrix, gate weight, conflict score, etc. This is **secondary**—needed for verification and debugging but not for baseline observation.

By placing the primary feed in a **dedicated zone** and the telemetry in **distinct panels**, the operator can:
- Focus on the video when observing behavior.
- Glance at telemetry when verifying a prediction.
- Avoid the cognitive load of parsing overlays on the video.

### 1.2.3 Reducing Cognitive Load

**Cognitive load** refers to the mental effort required to process information (Sweller, 1988). Reducing load improves performance: the operator can attend to the task (e.g., assessing affect) rather than decoding a cluttered display. Segregation allows **selective attention**—the operator chooses when to consult telemetry. It also allows **spatial chunking**—the brain groups related information (e.g., all FAU values in one panel) rather than scanning the entire display. The result is a more usable, less fatiguing interface.

---

# Part II — Practical Implementation (The Segregated Architecture)

## 2.1 The Dual-Zone Visual Layout

The XAI Dashboard is implemented as a single **1280×720** OpenCV window with **explicit structural separation** of the output visual layer into two conceptual zones.

### 2.1.1 Zone 1: The Primary Feed

**Zone 1** is dedicated solely to the **raw video footage** captured by the camera and the **recognized human state** in immediate context. It comprises:

- **Video region:** The live 640×480 camera feed, placed at coordinates (320, 90) on the canvas. Sci-fi-style corner brackets frame the video. Optional overlays (face bounding box, micro-expression banner) may appear on the video when relevant, but the video itself is kept as clean as possible.

- **State overlay:** The recognized state (e.g., "Sarcasm," "Panic," "Frustration") is **prominently displayed** in the top header, centered and large (font scale 1.1). The header is positioned directly above the video, providing **immediate baseline context**—the operator sees the state and the subject in the same visual sweep. The state color indicates source: green for neural, yellow for rule override.

- **Minimal header text:** Confidence percentage and logic source (Neural, Override, Posture-Override) appear in the header's right side. This provides essential context without cluttering the video.

The primary feed zone is kept **free of dense logical text**. No rule conditions, no conflict formulas, no gate equations—only the state, confidence, and source. The operator obtains immediate context: "What is the system saying?" and "Where did it come from?"

### 2.1.2 Zone 2: The Telemetry Pointers

**Zone 2** is a **dedicated secondary region** (or distinct graphical panels) strictly for **detailed pointers**. It includes:

- **Left panel (Visual Cortex):** Real-time FAU progress bars (FAU12, FAU6, FAU4) with numeric values; Posture Matrix (Lean, Shoulders Slumped, Asymmetry); Gestures (Finger Tapping, Hand-to-Face); control flag warnings (RULES: BYPASSED, VISUAL SENSOR: OFFLINE, AUDIO SENSOR: OFFLINE).

- **Right panel (Audio Cortex):** VU meter for intensity; jitter bar; AV sync delay slider.

- **Bottom panel (Fusion Engine):** Gate weight slider (Audio ↔ Visual dominant); Conflict Engine bar; FPS; temporal buffer status; smoothing status.

These panels occupy fixed regions (Left: 10–300 px; Right: 980–1270 px; Bottom: 580–710 px) and **do not overlay** the video. The operator can consult them when verifying a prediction or debugging, without obscuring the primary feed.

---

## 2.2 The "Why" via Terminal Reasoning

### 2.2.1 Deliberate Design Choice

The graphical interface is **deliberately kept clean** of dense logical text. Overlaying the exact reasoning—e.g., "Override: Hiding Stress (Neural=Neutral, rigidity=0.88, jitter=0.72)"—on the video would clutter the display and increase cognitive load. Instead, the **exact reasoning** is routed to the **system terminal**.

### 2.2.2 Terminal as Auditable Diagnostic Log

The terminal provides a **structured, auditable diagnostic log**. When a rule override occurs, the system can log:
- The rule that fired (e.g., "Hiding Stress").
- The neural prediction that was overridden (e.g., "Neural=Neutral").
- The telemetry that triggered the rule (e.g., "rigidity=0.88, jitter=0.72").

When the logic source changes, the system can log:
- The new state and source (e.g., "Logic: Posture-Override → Panic").
- The conflict score and gate weight when relevant.

The `state_container` holds `logic_source`, `conflict_score`, `neural_state`, and `gate_weight`—all of which can be printed to the terminal. This creates a **timestamped audit trail** that supports offline analysis, debugging, and regulatory compliance. The graphical interface shows the *current* state and telemetry; the terminal provides the *persistent* record of the "why."

---

## 2.3 Interactive Presentation Toggles ("God Mode")

The UI thread supports **interactive toggles** that allow the operator to control the system's behavior and visualization during live demonstration or debugging. These are mapped to keyboard keys:

| Key | Function | Effect |
|-----|----------|--------|
| **r** | Rule Bypass | Toggles `use_rules`. When OFF, the Symbolic Reasoning Engine is bypassed; the neural prediction is returned directly. Enables ablation: compare neural-only vs. neural+rules. |
| **v** | Visual Blindfold | Toggles `blind_v`. When ON, the visual stream is zeroed before the fusion head. The model receives only audio. Tests audio-only performance. |
| **a** | Audio Blindfold | Toggles `blind_a`. When ON, the audio stream is zeroed. The model receives only visual. Tests visual-only performance. |
| **l** | Landmark Meshes | Toggles `show_landmarks`. When ON, MediaPipe Face Mesh and Pose landmarks are drawn on the video. Shows the geometric data used for FAU and posture. |
| **c** | DINOv2 ROI PIP | Toggles `show_roi`. When ON, a grayscale face crop (112×112) is overlaid in the bottom-right of the video. Shows the region passed to DINOv2. |
| **q** / **ESC** | Quit | Exits the application. |

These toggles provide **live control** over the pipeline without restarting. They support demonstrations (e.g., "Watch what happens when we blindfold the visual sensor") and debugging (e.g., "Does the rule fire when we expect?").

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why output the detailed logical reasoning to the terminal instead of overlaying it on the video feed?

### Brief

Overlaying dense reasoning on the video would clutter the primary feed and increase cognitive load. The terminal provides a persistent, scrollable, auditable log that does not obscure the subject. The graphical interface stays clean for observation; the terminal serves verification and debugging.

### Detailed

**Cognitive load:** The video feed is the operator's primary focus. Overlaying text such as "Override: Hiding Stress (Neural=Neutral, rigidity=0.88, jitter=0.72)" would force the operator to read and parse while observing the subject. This splits attention and degrades both observation and verification. By routing reasoning to the terminal, the operator can (a) observe the video without distraction, and (b) consult the terminal when needed—e.g., after a prediction, to verify why it occurred.

**Persistence and auditability:** The terminal maintains a **scrollable history**. Overlays on the video are ephemeral—they disappear when the state changes. The terminal log persists, enabling post-hoc review. For regulatory or clinical audit, a timestamped log of reasoning is required; the terminal provides this. The graphical interface shows the *current* state; the terminal provides the *historical* record.

**Spatial separation:** The terminal is typically on a different part of the screen (or a separate window) from the video. The operator can position the terminal for comfortable side-by-side viewing. Overlays are constrained to the video region and cannot be repositioned.

### Comprehensive

The design reflects a **separation of concerns** between **real-time observation** and **verification/audit**. The graphical interface is optimized for the former: minimal overlay, clear state display, telemetry in dedicated panels. The terminal is optimized for the latter: dense, structured, persistent. Combining them would compromise both—the video would be cluttered, and the log would be ephemeral. The terminal also supports **automation**: logs can be piped to files, parsed by scripts, or integrated with external monitoring systems. Overlays cannot be easily captured or processed. For an XAI diagnostic tool, the terminal log is the **audit trail**; the graphical interface is the **operational view**. Both are necessary; they serve different purposes and are best kept separate.

---

## Q2: How does the separation of the raw video feed from the detailed facial/body pointers enhance the usability of the system?

### Brief

Separation allows the operator to focus on the video when observing behavior and to consult telemetry when verifying predictions. It reduces cognitive load by avoiding overlay clutter and enables spatial chunking—related telemetry is grouped in panels. The result is a more usable, less fatiguing interface.

### Detailed

**Selective attention:** The operator can choose when to attend to telemetry. During baseline observation (e.g., watching the subject speak), the video is sufficient. When a prediction appears (e.g., "Panic"), the operator can glance at the left panel to verify: shoulders_raised=True, Self_Touching_Hands=True. The separation allows this **glance-and-verify** workflow without forcing continuous parsing of overlays.

**Reduced clutter:** Overlaying FAU bars, posture flags, and gestures on the video would obscure the subject's face and body. The separation keeps the video clean—only optional overlays (landmarks, ROI PIP) when explicitly enabled. The telemetry is in fixed panels where it does not interfere with observation.

**Spatial chunking:** The brain groups related information. Placing all FAU values in one panel, all posture metrics in another, and all fusion data in a third allows the operator to quickly locate the relevant telemetry. With overlays, the operator would scan the entire video to find each value.

**Reduced fatigue:** A cluttered display increases mental effort and eye strain. A segregated layout reduces both. The operator can sustain longer sessions without fatigue, improving reliability in extended use (e.g., clinical assessment, research data collection).

### Comprehensive

Usability is measured by **efficiency** (time to complete a task), **accuracy** (correct interpretation of predictions), and **satisfaction** (operator comfort). Separation enhances all three. **Efficiency:** The operator finds telemetry faster when it is grouped in panels. **Accuracy:** A clean video allows accurate observation of the subject; telemetry in panels allows accurate verification of predictions. **Satisfaction:** Less clutter reduces stress and fatigue. The design follows established UI principles: **progressive disclosure** (show essential info first; details on demand) and **information scent** (group related items so users know where to look). The raw video is the "essential" view; the telemetry panels are "details on demand." This hierarchy matches the operator's workflow: observe first, verify when needed.

---

## Q3: How do the interactive toggles (like the Visual/Audio blindfolds) mathematically prove the functionality of the Gated Multimodal Fusion (GMF) layer during a live demonstration?

### Brief

When the visual blindfold is activated, the visual stream is zeroed; the gate should shift toward audio ($g \to 0$). When the audio blindfold is activated, the audio stream is zeroed; the gate should shift toward visual ($g \to 1$). Observing the gate weight in the Fusion Engine panel during these toggles provides direct evidence that the GMF layer is responding to modality availability.

### Detailed

**GMF behavior:** The gate $g$ is computed as $g = \sigma(W_{\text{gate}} \cdot [h_{\text{visual}}; h_{\text{audio}}] + b_{\text{gate}})$. When $h_{\text{visual}}$ is zeroed (blindfold), the gate input changes; the learned weights $W_{\text{gate}}$ should produce a lower $g$ (favoring audio). When $h_{\text{audio}}$ is zeroed, $g$ should increase (favoring visual). The fused representation $h_{\text{fused}} = g \cdot h_{\text{visual}} + (1-g) \cdot h_{\text{audio}}$ thus shifts toward the non-zeroed modality.

**Live demonstration:** With the system running, activate the visual blindfold (key `v`). The gate weight slider in the Fusion Engine panel should move toward "AUDIO DOMINANT" ($g$ decreases). The prediction may change (e.g., from a visual-dominant "Happy" to an audio-influenced state). Activate the audio blindfold (key `a`). The gate should move toward "VISUAL DOMINANT" ($g$ increases). Deactivate both; the gate should return to an intermediate value. This **observable behavior** proves that (1) the blindfolds affect the input to the fusion head, (2) the gate responds to the change, and (3) the fused output shifts accordingly.

**Mathematical proof:** The demonstration does not prove the *correctness* of the gate (that would require labeled data and evaluation metrics). It proves the **functionality**—that the gate is computed from both modalities and that zeroing one modality changes the gate. This is sufficient for a live demo to show that the GMF layer is operational and that the system adapts to modality availability.

### Comprehensive

The toggles provide **controlled ablation** in real time. Ablation—removing or zeroing a component and observing the effect—is a standard method for validating that a component contributes to the output. The blindfolds implement ablation at the input level: they zero one modality before it reaches the fusion head. The gate weight is the **observable output** of the GMF layer; it is displayed in the Fusion Engine panel. By toggling blindfolds and watching the gate, the demonstrator shows a **causal chain**: blindfold → zeroed input → changed gate input → changed $g$ → changed $h_{\text{fused}}$. This is a form of **sanity check**—if the gate did not change when a modality was zeroed, the GMF would be non-functional. The demonstration also has **pedagogical value**: it makes the abstract concept of "gated fusion" concrete. The audience sees the slider move and understands that the system is "listening" to both modalities and adjusting its reliance based on what it receives. For a systems engineering thesis, this live demonstration is evidence that the implementation matches the design: the GMF layer is not a black box; its behavior is observable and verifiable through the toggles.
