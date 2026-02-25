# 05 — Visual Expert (DINOv2)

## Chapter Overview

This document details the use of Meta's **DINOv2** Vision Transformer as the visual backbone of RHNS v2.0. It comprises three sections: (1) theoretical background on Vision Transformers and self-supervised learning; (2) practical implementation of the visual pipeline; and (3) an FAQ addressing anticipated defense queries.

---

# Part I — Theoretical Background (Vision Transformers & Self-Supervised Learning)

## 1.1 The Shift from CNNs to Vision Transformers

### 1.1.1 Convolutional Inductive Bias

**Convolutional Neural Networks (CNNs)** dominated computer vision for a decade. Their inductive bias—local connectivity and weight sharing—makes them efficient for capturing **local patterns** (edges, textures, small shapes). However, the receptive field of a convolution is limited; stacking layers increases it gradually, but long-range dependencies (e.g., the relationship between eyes and mouth) require many layers or large kernels.

### 1.1.2 Self-Attention and Global Context

**Vision Transformers (ViTs)** treat an image as a sequence of patches and apply **self-attention** across all patches. Each patch can attend to every other patch in a single layer. This enables **global context** from the first transformer block: the representation of the mouth region can directly incorporate information from the eyes, brows, and forehead. For affective computing, this is critical—facial expressions are holistic. A smile is not merely lip curvature; it involves cheek raising (FAU6), eye narrowing, and often brow position. The self-attention mechanism captures these **relational** structures (e.g., how the eyes relate to the mouth) rather than just local pixel textures.

### 1.1.3 ViT Architecture (Brief)

A ViT divides the image into fixed-size patches, linearly projects each patch to a token embedding, adds a learnable [CLS] token, applies positional embeddings, and processes the sequence through transformer encoder layers. The [CLS] token aggregates information from all patches via attention and serves as the image-level representation for downstream tasks.

---

## 1.2 Self-Supervised Learning (DINO)

### 1.2.1 Definition: DIstillation with NO Labels

**DINO** (Caron et al., 2021) stands for **DIstillation with NO labels**. It is a self-supervised learning framework that trains a Vision Transformer (or CNN) without manual labels. The key idea: a **student** network is trained to match the output of a **teacher** network on different augmented views of the same image. The teacher is a momentum-updated exponential moving average (EMA) of the student. The objective is a cross-entropy loss between the student's softmax output and the teacher's softmax output over the same image under different augmentations. No labels are used—the "supervision" comes from the consistency of representations across views.

### 1.2.2 Why Self-Supervised for Facial Feature Extraction

**Domain-specific supervised models** (e.g., VGG-Face, DeepFace) are trained on large face datasets (e.g., VGGFace2) with **identity labels**. They learn to maximize identity discrimination—features that distinguish one person from another. Such features can be overfit to identity-specific cues (e.g., facial structure, skin texture) rather than **expression** or **affective state**. For emotion recognition, identity is a confound; we want features that generalize across identities and capture expression dynamics.

**Self-supervised foundational models** like DINOv2 are trained on diverse, unlabeled images (e.g., ImageNet, LVD-142M). They learn **generic visual representations**—semantic structure, object parts, spatial relationships—without being tied to a specific supervised task. When applied to faces, these representations capture appearance and structure in a way that **generalizes** across identities, lighting, and pose. They are less prone to overfitting to identity-specific artifacts and more suitable for transfer to affective tasks (e.g., fine-tuning or using as frozen features). DINOv2, trained at scale with improved regularization, produces high-quality, robust embeddings that serve as a strong backbone for downstream tasks.

---

## 1.3 Patch Embeddings

### 1.3.1 Mathematical Formulation

For an image of height $H$ and width $W$, and a patch size $P \times P$, the image is divided into a grid of patches. The number of patches is:

$$
N = \frac{H \times W}{P^2} = \frac{H}{P} \times \frac{W}{P}
$$

Each patch is a contiguous $P \times P \times C$ region (where $C=3$ for RGB). The patches are flattened and linearly projected to a $d$-dimensional embedding space:

$$
\mathbf{z}_i = \mathbf{E} \cdot \text{flatten}(\mathbf{x}_i) + \mathbf{e}_{\text{pos},i}
$$

where $\mathbf{E} \in \mathbb{R}^{d \times (P^2 C)}$ is the patch embedding matrix, and $\mathbf{e}_{\text{pos},i}$ is the positional embedding for patch $i$. A [CLS] token $\mathbf{z}_0$ is prepended and updated through the transformer layers. The final output is the representation of the [CLS] token, which serves as the image-level embedding.

### 1.3.2 DINOv2 ViT-S/14

For **DINOv2 ViT-S/14** (`dinov2_vits14`):
- **Input:** $224 \times 224 \times 3$
- **Patch size:** $P = 14$
- **Number of patches:** $N = \frac{224}{14} \times \frac{224}{14} = 16 \times 16 = 256$
- **Sequence length:** $256 + 1 = 257$ (256 patches + 1 [CLS] token)
- **Embedding dimension:** $d = 384$

The final 384-D latent vector $h_{\text{visual}} \in \mathbb{R}^{384}$ is the [CLS] token output after the final transformer layer (optionally L2-normalized, as in `x_norm_clstoken`).

---

# Part II — Practical Implementation (The RHNS v2.0 Visual Pipeline)

## 2.1 Face ROI Isolation

### 2.1.1 Detection and Bounding Box

The pipeline uses **MediaPipe Face Mesh** to detect the face and obtain 468 facial landmarks. The first detected face is used. The bounding box is computed from the landmark extrema:

$$
x_{\min} = \min_i x_i, \quad y_{\min} = \min_i y_i, \quad x_{\max} = \max_i x_i, \quad y_{\max} = \max_i y_i
$$

where $(x_i, y_i)$ are the pixel coordinates of landmark $i$. A **20% margin** is added to include context (e.g., hairline, ears):

$$
x_{\min}' = \max(0, x_{\min} - 0.2 \cdot w_{\text{box}}), \quad x_{\max}' = \min(W, x_{\max} + 0.2 \cdot w_{\text{box}})
$$

and similarly for $y$, where $w_{\text{box}} = x_{\max} - x_{\min}$, $h_{\text{box}} = y_{\max} - y_{\min}$.

### 2.1.2 Cropping and Resizing

The face region is cropped from the BGR frame using OpenCV:

```python
face_crop = frame_bgr[y_min_i:y_max_i, x_min_i:x_max_i]
```

The crop is resized to **224×224** using bilinear interpolation (`cv2.INTER_LINEAR`). This fixed resolution is required by DINOv2, which expects a fixed input size.

### 2.1.3 ImageNet Normalization

The resized crop is converted from BGR to RGB, scaled to $[0, 1]$, and normalized with ImageNet statistics:

$$
\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} / 255 - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
$$

where $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$ (channel-wise). The tensor is then permuted to `[1, C, H, W]` for the model input.

---

## 2.2 Latent Vector Generation

### 2.2.1 Forward Pass

The normalized tensor is passed through the DINOv2 model on the configured device (`mps` on Apple Silicon, `cuda` on NVIDIA, `cpu` otherwise):

```python
with torch.no_grad():
    out = self._dinov2(tensor)
```

### 2.2.2 CLS Token Extraction

The output may be a tensor of shape `(B, 1+N, C)` or a dictionary. The system extracts the [CLS] token representation:

- If `out` is a dict: `feat = out.get("x_norm_clstoken") or out.get("x") or next(iter(out.values()))`
- If `out` is a tensor: `feat = out[:, 0, :]` (first token is [CLS])

The result is squeezed to `(384,)` and converted to CPU numpy as float32.

### 2.2.3 Fallback

If no face is detected, or if the crop is invalid, the system returns `np.zeros(384, dtype=np.float32)` to avoid downstream errors.

---

## 2.3 Hardware Engineering (The MPS Fallback)

### 2.3.1 Apple Silicon and Metal Performance Shaders

On **Apple Silicon (M1/M2/M3)**, PyTorch uses the **Metal Performance Shaders (MPS)** backend for GPU acceleration. MPS maps PyTorch operations to Metal's GPU compute API, enabling efficient execution of neural network workloads on the integrated GPU.

### 2.3.2 The Unsupported Operator: `aten::upsample_bicubic2d.out`

The DINOv2 model (and many ViT implementations) includes operations that are **not natively supported** by MPS. Specifically, the **`aten::upsample_bicubic2d.out`** operator—used for bicubic interpolation when resizing feature maps within the model (e.g., in positional embedding or patch embedding stages)—is not implemented in the Metal backend. When PyTorch encounters this operator during the forward pass, it raises a runtime error:

```
RuntimeError: The following operation(s) are not supported:
  aten::upsample_bicubic2d.out
```

### 2.3.3 The Graceful Fallback

The environment variable **`PYTORCH_ENABLE_MPS_FALLBACK=1`** instructs PyTorch to **fall back to CPU** for unsupported operations. When the forward pass reaches `upsample_bicubic2d`, PyTorch:

1. Copies the required tensors from MPS to CPU (if needed).
2. Executes the operation on CPU.
3. Copies the result back to MPS (if needed).
4. Continues the rest of the forward pass on MPS.

**Effect:** The heavy matrix multiplications (attention, MLP blocks) remain on the GPU; only the unsupported resize operation runs on CPU. The performance impact is limited because the resize is a small fraction of the total compute. The alternative—running the entire model on CPU—would be significantly slower.

### 2.3.4 Usage

The variable must be set **before** importing PyTorch:

```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

Or at launch:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py
```

---

# Part III — Comprehensive FAQ (Anticipated Defense Queries)

## Q1: Why use DINOv2 (a general-purpose self-supervised ViT) instead of a domain-specific facial recognition model like VGG-Face or DeepFace?

### Brief

DINOv2 provides **generalized visual representations** that transfer well to affective tasks without identity overfitting. VGG-Face and DeepFace are trained for identity recognition; their features emphasize person-specific cues that can confound expression recognition and do not generalize as well across unseen identities.

### Detailed

**VGG-Face / DeepFace:** These models are trained on large face datasets (e.g., VGGFace2) with **identity labels**. The objective is to maximize identity discrimination—features that distinguish one person from another. The learned representations are optimized for "who" rather than "how they feel." When applied to emotion recognition, they can:
- Overfit to identity-specific cues (e.g., facial structure, skin texture) that correlate with identity in the training set but not with expression.
- Underperform on unseen identities (out-of-distribution generalization).
- Require fine-tuning on emotion-labeled data to adapt, which may reintroduce overfitting if the emotion dataset is small.

**DINOv2:** Trained with **self-supervised learning** on diverse, unlabeled images (e.g., ImageNet, LVD-142M). It learns generic visual structure—semantic parts, spatial relationships, appearance—without identity supervision. When applied to faces:
- The representations capture appearance and structure in a way that **generalizes** across identities.
- The model has not been optimized to distinguish identities; it is less prone to identity-specific artifacts.
- The 384-D embedding is a compact, semantically rich representation suitable for downstream fusion (e.g., with audio) without requiring large emotion-specific fine-tuning.

For RHNS v2.0, the goal is **affective state** recognition, not identity. DINOv2's general-purpose features, combined with the fusion head's training on CREMA-D (6-class emotion), provide a better balance of generalization and task-specific performance than identity-focused models.

### Comprehensive

The choice reflects a **transfer learning** and **domain shift** argument. **Transfer learning:** Identity models and emotion models require different feature spaces. Identity features emphasize invariant (across expression) cues; emotion features emphasize variant (across expression) cues. A model trained for identity may suppress expression-relevant features. DINOv2, trained without identity supervision, retains a broader feature space that includes expression-relevant structure. **Domain shift:** CREMA-D actors may not overlap with the identity model's training set. Identity features can be unreliable on out-of-distribution faces. DINOv2's general-purpose features are more robust to such shift. **Empirical:** The RHNS architecture achieves base-emotion classification with DINOv2 + fusion head; no ablation study was conducted with VGG-Face, but the theoretical and transfer-learning arguments support the DINOv2 choice. **Scalability:** DINOv2 is actively maintained; it can be updated with newer self-supervised versions (e.g., DINOv2-giant) if needed. Identity models are often tied to specific architectures and datasets with fixed licenses.

---

## Q2: How does the system handle the dimensionality reduction from a raw 640×480 webcam frame to a 384-D latent vector?

### Brief

The frame is reduced in stages: (1) face detection and cropping to a variable-size ROI; (2) resize to 224×224; (3) patch embedding to 256 tokens of 384-D each; (4) transformer processing to aggregate into a single 384-D [CLS] token. The reduction is learned by the pretrained DINOv2 model, not hand-designed.

### Detailed

**Stage 1 — Spatial reduction (face ROI):** The full 640×480 frame (307,200 pixels) is processed by MediaPipe Face Mesh. The face bounding box is extracted; typically the face occupies 60×80 to 120×160 pixels. Cropping reduces the region to ~5,000–20,000 pixels.

**Stage 2 — Fixed resize (224×224):** The crop is resized to 224×224 = 50,176 pixels. This is a fixed input size for DINOv2; aspect ratio may be distorted.

**Stage 3 — Patch embedding:** The 224×224 image is split into 16×16 = 256 patches of 14×14 pixels each. Each patch is flattened (14×14×3 = 588 values) and linearly projected to 384-D. The result is 256 tokens of 384-D.

**Stage 4 — Transformer aggregation:** The 256 patch tokens plus 1 [CLS] token are processed through 12 transformer layers. Self-attention allows the [CLS] token to aggregate information from all patches. The final [CLS] token is the 384-D output—a single vector that summarizes the entire face region.

**Information flow:** The reduction is **lossy** but **learned**. The pretrained DINOv2 has learned to compress spatial information into a compact representation that preserves semantic structure. The 384-D vector does not encode every pixel; it encodes a summary of appearance, expression, and context that is useful for downstream tasks.

### Comprehensive

The dimensionality reduction is a form of **learned compression**. The raw frame has 307,200 × 3 = 921,600 values (before normalization). The 384-D output is a 2,400× compression ratio. This compression is achieved by:
1. **Spatial downsampling** (crop + resize): Reduces resolution; discards background and non-face regions.
2. **Patch-based representation:** Groups pixels into semantic units (patches); reduces redundancy.
3. **Transformer aggregation:** The [CLS] token attends to all patches and produces a single summary. The attention mechanism learns which patches contribute to the final representation.

The key insight is that the reduction is **task-relevant**. DINOv2 was trained (via self-supervision) to produce representations that preserve semantic structure. For affective computing, the 384-D vector captures expression-relevant information (e.g., mouth shape, brow position, eye openness) while discarding identity-specific or noisy details. The fusion head learns to map this 384-D representation to emotion classes; the pretrained embedding provides a strong inductive bias for that mapping.

---

## Q3: Explain the exact hardware limitation on Apple Silicon that necessitates the PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable during visual inference.

### Brief

The DINOv2 forward pass uses the `aten::upsample_bicubic2d.out` operator for bicubic interpolation. This operator is not implemented in PyTorch's Metal Performance Shaders (MPS) backend for Apple Silicon. Without the fallback, the operator raises a runtime error. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` instructs PyTorch to execute unsupported ops on CPU while keeping the rest on MPS.

### Detailed

**Operator:** `aten::upsample_bicubic2d.out` performs a 2D bicubic upsampling of a tensor. It is used in the DINOv2 (and many ViT) implementations for operations such as:
- Resizing positional embeddings to match the spatial resolution of feature maps.
- Interpolating feature maps during patch embedding or in attention modules.

**MPS backend:** PyTorch's MPS backend maps PyTorch ops to Metal shaders. Metal provides a subset of operations; not all PyTorch ops have Metal implementations. The `upsample_bicubic2d` op is one of the missing ops. When the MPS backend encounters it, it raises:

```
RuntimeError: The following operation(s) are not supported:
  aten::upsample_bicubic2d.out
```

**Cause:** The Metal API does not expose a direct equivalent of bicubic interpolation in the format PyTorch expects, or the PyTorch-MPS integration has not yet implemented this op. The MPS backend is under active development; new ops are added over time.

**Fallback mechanism:** When `PYTORCH_ENABLE_MPS_FALLBACK=1` is set, PyTorch registers a fallback for unsupported ops. At runtime, when the MPS backend encounters an unsupported op, it:
1. Dispatches the op to the CPU implementation.
2. Transfers tensors between MPS and CPU as needed.
3. Returns the result to the MPS stream.

The rest of the computation (matrix multiplies, attention, etc.) remains on MPS. Only the unsupported op runs on CPU.

### Comprehensive

The limitation is **architectural** rather than fundamental. Apple Silicon (MPS) and NVIDIA (CUDA) implement different backend APIs. PyTorch maintains separate implementations for each; some ops are implemented on CUDA but not yet on MPS. The `upsample_bicubic2d` op is used in:
- DINOv2's patch embedding or positional encoding (e.g., for variable-resolution inputs).
- Other ViT implementations that use bicubic interpolation for spatial operations.

The fallback is a **graceful degradation** strategy: the system continues to run on Apple Silicon with minimal code changes. The performance cost of running one or a few ops on CPU is negligible compared to the cost of running the entire model on CPU. The alternative—implementing a custom MPS kernel for `upsample_bicubic2d` or patching DINOv2 to use a different resize op—would require significant engineering. The environment variable provides a practical solution for deployment on M-series Macs. As PyTorch's MPS backend matures, this op may be added natively, and the fallback would no longer be necessary.
