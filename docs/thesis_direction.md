# Thesis Direction: From Diagnostic Findings to Research Contribution

## 1. The Diagnostic Finding

Linear probing on frozen REVE embeddings reveals a striking pattern when the representation strategy is varied:

| Representation | Dim | Binary Acc | 9-Class Acc | Evaluation |
|----------------|-----|-----------|-------------|------------|
| Attention pooling (`pool`) | 512 | 55.76% | ~21% | Cross-subject, 10s windows |
| Flattened 4-D output (`nopool_flat`) | 168,960 | 66.11% | 38.02% | Cross-subject, 10s non-overlapping |
| Flattened 4-D output (`nopool_flat`) | ~507,000 | **82%** | **71%** | Cross-subject, 30s full stimulus |

All results are validated on held-out subjects (10-fold cross-subject CV). The encoder is entirely frozen — no parameter is updated.

**Key observation:** REVE's internal spatial-temporal representation `(B, 30, H, 512)` encodes sufficient information for high cross-subject emotion recognition. However, REVE's attention pooling — which compresses this representation to 512 dimensions — discards the majority of that information. The gap between 512-D pooled (56%/21%) and full-stimulus flat (82%/71%) demonstrates that the bottleneck is in the aggregation mechanism, not in the encoder itself.

## 2. Why the 30s Flat Result Is Diagnostic, Not a Baseline

The 82%/71% result from 30s flat probing, while validated on unseen subjects, must be interpreted carefully:

### 2.1 Stimulus fingerprinting, not emotion recognition

THU-EP uses 28 fixed stimuli. In cross-subject CV, the **same 28 stimuli appear in every fold** — only the subjects change. With 30s windows (entire stimulus duration) and ~507K features, the linear classifier has access to the complete temporal profile of each stimulus response. This enables a classification strategy based on **stimulus template matching**: "this EEG response has temporal structure consistent with stimulus #7 (labelled Anger), as observed across training subjects."

This is not emotion recognition in the general sense. The classifier does not learn "what anger looks like in EEG." It learns "what stimulus #7 looks like in EEG." These are different:

- **Stimulus recognition** generalises to new subjects viewing the same stimuli (cross-subject, same-stimulus — the current protocol).
- **Emotion recognition** should generalise to new subjects viewing new stimuli that evoke the same emotion (cross-subject, cross-stimulus).

Evidence that stimulus fingerprinting drives the 30s result:

1. **Accuracy scales monotonically with window length** — 5s (57.5%) → 8s (61.1%) → 10s (66.1%) → 30s (82%). Longer windows capture more of the stimulus-locked temporal structure, making template matching easier. If the classifier were learning emotion-invariant features, window length should matter less once a sufficient temporal context is captured.

2. **The parameter-to-sample ratio is extreme.** With ~507K features and ~1,900 training samples (28 stimuli × ~68 train subjects per fold), the linear head has ~1M parameters for binary classification — a 530:1 parameter-to-sample ratio. The classifier has enough capacity to memorise a separate template for each of the 28 stimuli across training subjects.

3. **No windowing = one sample per stimulus per subject.** The classifier cannot learn sub-stimulus temporal dynamics. It receives the entire 30s response as a monolithic feature vector. Any classification signal comes from the overall temporal profile (stimulus-locked ERPs, sustained oscillatory patterns), not from affective features that vary across stimuli of the same emotion.

### 2.2 Impracticality

A 507K-dimensional linear classifier is not a learned representation. It cannot be transferred, interpreted, or deployed. It is a diagnostic tool that answers one question: "does the encoder contain the information?" The answer is yes.

## 3. The Actual Problem

The findings establish a clear structure:

```
Encoder output (30 × H × 512)
        │
        ├──[flatten]──→ 507K-D ──→ LP ──→ 82% / 71%  (information present)
        │
        └──[attention pool]──→ 512-D ──→ LP ──→ 56% / 21%  (information lost)
```

REVE's attention pooling was pre-trained via masked-patch reconstruction. Its `cls_query_token` learns to summarise spatial-temporal patterns that are useful for reconstructing masked EEG patches — primarily local waveform morphology and channel layout. It has no incentive to preserve:

- **Cross-channel coherence patterns** (e.g., frontal alpha asymmetry, correlated frontal-temporal activation) that index emotional valence.
- **Slow temporal dynamics** (e.g., late positive potentials, sustained oscillatory changes) that distinguish emotional categories.
- **Subject-invariant features** — the reconstruction objective operates within single subjects and has no mechanism to prioritise features that transfer across individuals.

The result is a 512-D summary that is informative for EEG reconstruction but agnostic to emotional content.

## 4. The Research Gap

The finding creates a clear, specific gap:

> REVE's encoder captures emotion-discriminative EEG features that generalise across subjects, but its pre-trained aggregation mechanism fails to preserve them. A task-specific mechanism is needed that compresses the spatial-temporal representation into a compact embedding while retaining emotion-relevant information and discarding subject-specific confounders.

No existing work addresses this for EEG foundation models. Prior contrastive learning work in EEG emotion recognition (e.g., CLISA, CDCL) operates on traditional feature extractors (CNNs, shallow networks), not on foundation model representations. The specific challenge of learning an emotion-aware aggregation on top of a large pre-trained encoder is uncharted.

## 5. Revised Thesis Architecture

### 5.1 Overview

The proposed method replaces REVE's pre-trained attention pooling with a **learned, contrastive-aligned aggregation** that compresses `(B, 30, H, 512)` into a compact embedding optimised for cross-subject emotion discrimination.

```
EEG window (30 channels × T samples)
        │
   REVE encoder (frozen or LoRA-adapted)
        │
   (B, 30, H, 512)  ←── full spatial-temporal representation
        │
   ┌────┴────┐
   │  Learnable aggregation / projection head  │
   │  (B, 30, H, 512) → (B, D), D ∈ {128, 256, 512}  │
   └────┬────┘
        │
   ┌────┴────────────────┐
   │                     │
   Projection head       Classification head
   (B, D) → (B, P)      (B, D) → (B, N_classes)
   │                     │
   SupCon loss           Cross-entropy loss
   │                     │
   └────────┬────────────┘
            │
     L = α · L_CE + β · L_SupCon
```

### 5.2 Why contrastive learning specifically

The supervised contrastive (SupCon) loss provides a property that cross-entropy alone cannot: **explicit inter-sample geometric structure.** Cross-entropy optimises each sample's class probability independently — it has no mechanism to enforce that same-emotion embeddings from different subjects are nearby, or that different-emotion embeddings are separated. SupCon directly optimises for this:

- **Positive pairs:** same emotion, different subjects → pulled together. Forces the aggregation to find features that are consistent across subjects for a given emotion.
- **Negative pairs:** different emotions → pushed apart. Forces the aggregation to find features that discriminate emotions, not subjects.

This is precisely the inductive bias needed to solve the aggregation problem: learn to compress spatial-temporal features while preserving emotion and discarding subject identity.

### 5.3 Operating on the 4-D representation, not the 512-D pooled output

A critical design choice: the contrastive loss should operate on the 4-D encoder output, not on REVE's pre-computed 512-D pooled embedding. The pooled embedding has already lost the emotion-relevant information (as shown by the 56%/21% linear probing result). Applying contrastive learning to an already-impoverished representation cannot recover what was discarded during pooling. The learnable aggregation must have access to the full spatial-temporal tensor to select and compress the right features.

### 5.4 Window strategy

Use **10s non-overlapping windows** (3 per stimulus):

- Yields ~5,700 training samples (3 windows × 28 stimuli × ~68 train subjects per fold)
- Sufficient for SupCon — each batch of 64 generates ~2,000 pairwise comparisons
- Non-overlapping avoids the pseudo-replication issue identified in the linear probing ablation
- 10s provides enough temporal context for meaningful emotion dynamics without degenerating into stimulus fingerprinting

## 6. Evaluation Strategy

### 6.1 Cross-subject CV (standard)

10-fold cross-subject CV as in the linear probing baseline. Unseen subjects, same stimuli. This is the standard protocol in the field and enables comparison with prior work.

### 6.2 Cross-stimulus generalisation (implemented: `--generalization`)

A stricter evaluation protocol, implemented as the `--generalization` flag in all training scripts. Within the same 10-fold cross-subject CV, stimuli are additionally split per emotion category:

- **Train set:** train subjects × 2/3 of stimuli per emotion (2 out of 3 for most emotions; 3 out of 4 for Neutral)
- **Val set:** held-out subjects × remaining 1/3 of stimuli

The validation set thus contains **entirely unseen subjects AND entirely unseen stimuli.** The stimulus split is deterministic (seed 123), seeded-shuffled per emotion group to avoid positional bias, and fixed across all folds.

Concrete split (binary mode, 16 train / 8 test stimuli):

- Train: `[0, 1, 3, 4, 7, 8, 9, 10, 16, 17, 19, 21, 22, 24, 25, 27]`
- Test:  `[2, 5, 6, 11, 18, 20, 23, 26]` — one per emotion category

This tests whether the model has learned **emotion recognition** (generalises to new stimuli evoking the same emotion) rather than **stimulus recognition** (generalises to new subjects viewing the same stimuli).

Expected behaviour:

| Method | Cross-subject (same stimuli) | Cross-stimulus (new stimuli) |
|--------|-----------------------------|-----------------------------|
| Flat LP (30s) | 82% / 71% | **Severe degradation** — cannot fingerprint unseen stimuli |
| Pooled LP (512-D) | 56% / 21% | Similar or slightly worse — already near chance |
| Contrastive (proposed) | Target: 70–80% / 50–65% | **Moderate degradation** — if genuinely emotion-aware |

Cross-stimulus evaluation is where the contrastive contribution becomes most visible. The flat LP, which relies on stimulus templates, should collapse. The contrastive model, if it has learned emotion-invariant features, should degrade more gracefully.

## 7. Revised Research Questions

**RQ1** remains: *To what extent can REVE, as a frozen foundation model, support cross-subject EEG emotion recognition?*

Answer (from linear probing): REVE's encoder captures rich emotion-discriminative features that generalise across subjects (82% binary / 71% 9-class when the full representation is accessible). However, its pre-trained attention pooling discards most of this information, reducing performance to near-chance levels (56% / 21%). The answer to RQ1 is therefore nuanced: the encoder succeeds, but the standard interface fails.

**RQ2** refines to: *Can a supervised contrastive learning objective train a task-specific aggregation mechanism that recovers the emotion signal lost during pooling, producing a compact and subject-invariant embedding?*

The contribution is not "contrastive learning improves accuracy" in the generic sense. It is: **contrastive learning solves the specific aggregation problem** identified by the diagnostic ablation — learning to compress REVE's spatial-temporal representation into a compact embedding that retains emotion and discards subject identity, achieving in a principled way what the impractical 507K-D flat probe achieves by brute force.

## 8. Expected Contributions

1. **Diagnostic analysis of EEG foundation model representations for emotion recognition.** First systematic evaluation of how aggregation strategy (pooling vs. flat) affects cross-subject emotion recognition with frozen REVE. The finding that pooling discards most emotion information is novel and has implications for how foundation models are used downstream.

2. **Contrastive aggregation framework.** A method that replaces task-agnostic pooling with a contrastive-trained projection, specifically designed to produce subject-invariant emotion embeddings from foundation model representations.

3. **Cross-stimulus evaluation protocol.** Leave-stimuli-out CV as a more rigorous test of emotion generalisation, distinguishing stimulus recognition from emotion recognition.
