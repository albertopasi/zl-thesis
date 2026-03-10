# Baseline 1: Linear Probing on Frozen REVE Embeddings

**Evaluation protocol:** 10-fold cross-subject cross-validation (79 subjects, ~8 per test fold)

**Dataset:** THU-EP — see `excluded_data.md` for corruption details

**Architecture:** Frozen REVE encoder → representation extraction → linear classifier (cross-entropy)

**Training:** Adam (lr = 1e-3), batch size 64, max 80 epochs, early stopping on val accuracy (patience 15)

---

## 1. Experimental Design

### Representation strategies

REVE's transformer outputs a 4-D tensor of shape `(B, 30, H, 512)` — 30 EEG channels, H time patches (dependent on window length), 512 hidden dimensions. Three strategies are compared for converting this into a fixed-size vector for the linear head:

| Strategy | Notation | How | Embed dim (10s window) |
| ---------- | ---------- | ----- | ------------------------ |
| **Attention pooling** | `pool` | REVE's learnable `cls_query_token` aggregates all channels and time patches → `(B, 512)` | 512 |
| **No-pool, channel-mean** | `nopool_mean` | Mean over the 30-channel dim → `(B, H, 512)` → flatten | 5,632 |
| **No-pool, flatten** | `nopool_flat` | Flatten entire 4-D output → `(B, 30 × H × 512)` | 168,960 |

For the flat strategy, dimensionality scales with window size:

| Window | H (time patches) | Flat dim |
| -------- | ------------------- | ---------- |
| 5s (1000 samples) | 5 | 76,800 |
| 8s (1600 samples) | 8 | 122,880 |
| 10s (2000 samples) | 11 | 168,960 |

### Window and stride configurations

| Label | Window | Stride | Overlap | Windows per 30s stimulus |
| ------- | -------- | -------- | --------- | -------------------------- |
| w5s2 | 5s | 2s | 60% | 13 |
| w8s4 | 8s | 4s | 50% | 6 |
| w10s5 | 10s | 5s | 50% | 5 |
| w10s10 | 10s | 10s | 0% | 3 |

---

## 2. Binary Task (Positive vs. Negative)

Random chance: 50.00%

### 2.1 Effect of representation (fixed 10s window)

| Representation | Stride | Dim | Acc (mean ± std) | AUROC (mean ± std) | F1 (mean ± std) | Train Acc @ best | Final Train Acc |
| ---------------- | -------- | ----- | ------------------- | -------------------- | -------------------- | ------------------ | ----------------- |
| pool | 5s | 512 | 55.76% ± 1.67% | 0.5620 ± 0.0260 | 0.517 ± 0.072 | 57.90% | 59.76% |
| nopool_mean | 5s | 5,632 | 59.88% ± 1.47% | 0.6225 ± 0.0189 | 0.603 ± 0.035 | 65.43% | 67.65% |
| nopool_flat | 5s | 168,960 | 62.66% ± 1.71% | 0.6479 ± 0.0197 | 0.608 ± 0.054 | 85.72% | 94.57% |
| nopool_flat | 10s | 168,960 | **66.11% ± 1.65%** | **0.6967 ± 0.0216** | **0.648 ± 0.041** | 82.80% | 95.07% |

### 2.2 Effect of window size (flat representation)

| Window | Stride | Dim | Acc (mean ± std) | AUROC (mean ± std) | F1 (mean ± std) | Train Acc @ best | Final Train Acc |
| -------- | -------- | ----- | ------------------- | -------------------- | -------------------- | ------------------ | ----------------- |
| 5s | 2s | 76,800 | 57.54% ± 1.70% | 0.5955 ± 0.0257 | 0.551 ± 0.054 | 75.49% | 85.88% |
| 8s | 4s | 122,880 | 61.07% ± 2.20% | 0.6381 ± 0.0237 | 0.587 ± 0.068 | 80.78% | 91.97% |
| 10s | 5s | 168,960 | 62.66% ± 1.71% | 0.6479 ± 0.0197 | 0.608 ± 0.054 | 85.72% | 94.57% |
| 10s | 10s | 168,960 | **66.11% ± 1.65%** | **0.6967 ± 0.0216** | **0.648 ± 0.041** | 82.80% | 95.07% |

---

## 3. Nine-Class Task (9 Emotions)

Random chance: 11.11% (1/9)

| Representation | Window | Stride | Dim | Acc (mean ± std) | AUROC (mean ± std) | F1 (mean ± std) | Train Acc @ best | Final Train Acc |
| ---------------- | -------- | -------- | ----- | ------------------- | -------------------- | -------------------- | ------------------ | ----------------- |
| nopool_flat | 10s | 5s | 168,960 | 34.14% ± 3.10% | 0.7064 ± 0.0250 | 0.334 ± 0.033 | 89.46% | 93.41% |
| nopool_flat | 10s | 10s | 168,960 | **38.02% ± 2.05%** | **0.7568 ± 0.0241** | **0.374 ± 0.025** | 90.02% | 94.73% |

For reference, the previous iteration of this baseline with attention-pooled 512-D embeddings (10s/5s window) yielded 21.37% accuracy and 0.6244 AUROC — the unpooled flat representation nearly doubles the margin above chance.

---

## 4. Interpretation

### 4.1 REVE's attention pooling is a severe bottleneck for emotion recognition

The central finding of these experiments is that **compressing REVE's 4-D output into a 512-D attention-pooled summary discards the majority of emotion-relevant information.** The magnitude of the improvement when bypassing pooling is not incremental — it is a qualitative shift:

| Task | Pooled Acc | Flat Acc | Gain |
| ------ | ----------- | ---------- | ------ |
| Binary | 55.76% | 66.11% | +10.35 pp |
| 9-class | ~21.37% | 38.02% | +16.65 pp |

REVE's `cls_query_token` was pre-trained via masked-patch reconstruction — an objective that rewards summarising spatial layout and local waveform morphology for the purpose of reconstructing missing EEG patches. Emotional state, however, is encoded in cross-channel coherence, hemispheric asymmetries, and slow temporal dynamics — features that the reconstruction objective has no explicit incentive to preserve in the pooled summary. When the classifier has direct access to all 30 channels × H time patches × 512 dimensions, it can locate and exploit these spatial-temporal emotion signatures.

This result has a direct implication for RQ1: the question "can REVE overcome inter-subject variability?" is representation-dependent. At the pooled level, REVE barely exceeds chance on binary valence. At the unpooled level, it reaches nearly 70% AUROC on binary and 76% on 9-class — evidence that the frozen encoder genuinely captures emotion-relevant EEG structure, but this structure is inaccessible through the standard pooling interface.

### 4.2 Overfitting and the dimensionality–generalisation trade-off

Bypassing pooling introduces a dimensionality explosion (512 → 168,960). Combined with ~8,500 training windows (binary) or ~9,900 (9-class, overlapping) or ~5,100–5,900 (non-overlapping), this creates an extreme parameter-to-sample imbalance:

| Representation | Dim | Linear params (binary / 9-class) | Approx. train samples |
| ---------------- | ----- | ---------------------------------- | ---------------------- |
| pool | 512 | 1,026 / 4,617 | 8,500 |
| nopool_mean | 5,632 | 11,266 / 50,697 | 8,500 |
| nopool_flat | 168,960 | 337,922 / 1,520,649 | 5,100–9,900 |

The overfitting signature is unmistakable and scales monotonically with dimensionality:

| Representation | Train Acc @ best epoch | Val Acc @ best epoch | Gap |
| ---------------- | ---------------------- | --------------------- | ----- |
| pool (binary) | 57.90% | 55.76% | 2.14 pp |
| nopool_mean (binary) | 65.43% | 59.88% | 5.55 pp |
| nopool_flat (binary, w10s5) | 85.72% | 62.66% | 23.06 pp |
| nopool_flat (binary, w10s10) | 82.80% | 66.11% | 16.69 pp |
| nopool_flat (9-class, w10s5) | 89.46% | 34.14% | 55.32 pp |
| nopool_flat (9-class, w10s10) | 90.02% | 38.02% | 52.00 pp |

Several patterns emerge:

**The pool regime underfits.** With 512-D embeddings and ~1,000 parameters, the linear head barely overfits (2 pp gap). This is not a sign of good generalisation — it is a sign that there is so little information in the representation that even a linear classifier with 1,026 parameters cannot find more than a marginal signal. The bottleneck is informational, not architectural.

**The flat regime overfits heavily but also achieves the best validation metrics.** The 9-class flat configuration overfits by 52–55 pp, yet achieves 38% val accuracy vs. ~21% with pooling. Overfitting is not inherently bad — what matters is whether the additional capacity *also* improves generalisation, and here it clearly does. The model memorises training-subject patterns, but in doing so it also learns some cross-subject emotion structure that transfers. The gap between 82–90% train and 34–66% val represents the "subject identity tax": the fraction of the model's performance that is attributable to subject-specific confounders rather than universal emotion features.

**Validation loss diverges while validation accuracy plateaus.** This is the most diagnostically informative pattern. In the flat regime, val loss grows 3–5× over training (e.g., from ~30 to ~130 for 9-class w10s5), while val accuracy remains in a narrow range (30–40%). The model becomes progressively more *confident* in its predictions — both correct and incorrect — without becoming more *accurate*. This is classic miscalibration: the softmax probabilities sharpen toward one-hot distributions for both train and val, but the sharpening is only justified for training subjects where the patterns are genuine. For unseen subjects, the overconfident wrong predictions inflate cross-entropy loss while the argmax prediction stays roughly constant.

**AUROC is more stable than accuracy across epochs.** AUROC measures *ranking* quality independent of calibration. In the 9-class flat regime, val AUROC stabilises around 0.70–0.76 early in training and holds even as val loss triples. This confirms that the model's probability *ordering* — which classes it considers more likely — remains informative even when its probability *magnitudes* are catastrophically miscalibrated. The ranking captures genuine emotion structure in the embeddings; the calibration reflects subject-specific overfitting. This dissociation is strong evidence that a well-designed alignment objective (contrastive learning) can preserve and amplify the ranking signal while correcting the calibration.

### 4.3 Non-overlapping windows outperform overlapping windows

A consistent and initially counter-intuitive finding: increasing the stride to eliminate overlap improves generalisation, despite reducing the number of training samples:

| Config | Overlap | Windows/stim | Approx. train windows | Val Acc | Val AUROC |
| -------- | --------- | ------------- | ---------------------- | --------- | ----------- |
| Binary flat w10s5 | 50% | 5 | ~8,500 | 62.66% | 0.6479 |
| Binary flat w10s10 | 0% | 3 | ~5,100 | **66.11%** | **0.6967** |
| 9-class flat w10s5 | 50% | 5 | ~9,900 | 34.14% | 0.7064 |
| 9-class flat w10s10 | 0% | 3 | ~5,900 | **38.02%** | **0.7568** |

**Fewer samples, better performance.** This reveals that 50%-overlapping windows introduce correlated pseudo-replication: consecutive windows share half their temporal content, so the classifier repeatedly sees the same EEG segments in slightly shifted context. This inflates the apparent training set size without adding genuine informational diversity. The model exploits these intra-stimulus correlations to achieve high training accuracy, but the correlations are stimulus-specific and do not generalise.

Non-overlapping windows provide temporally independent segments. Each window contributes unique information, and the classifier cannot exploit inter-window redundancy. The result is a more honest training signal and better cross-subject transfer.

**Practical implication:** Non-overlapping windows (stride = window_size) should be the default sampling strategy for all subsequent methods, including contrastive learning where pair diversity is essential.

### 4.4 Window size matters — but only without pooling

Under attention pooling (512-D), window size had negligible impact: earlier experiments with 5s, 8s, and 10s windows yielded accuracy within ±0.9 pp (noise-level variation). This was expected, since pooling compresses all windows to the same 512-D bottleneck regardless of input length.

Without pooling, window size becomes a meaningful variable:

| Window (flat repr.) | Binary Acc | Binary AUROC |
| --------------------- | ----------- | ------------- |

| 5s/2s | 57.54% | 0.5955 |
| 8s/4s | 61.07% | 0.6381 |
| 10s/5s | 62.66% | 0.6479 |
| 10s/10s | 66.11% | 0.6967 |

The spread is now **8.57 pp** in accuracy — far beyond the ±1.7% fold-level standard deviation. Longer windows capture richer temporal dynamics (more complete affective transients, finer frequency resolution, longer-range dependencies), and the flat linear head can exploit these additional features because they are not compressed away. This suggests that the temporal structure of emotional EEG responses extends over several seconds, and that access to the full temporal representation is critical for emotion classification.

The 10s window captures a substantial portion of the 30s stimulus presentation, providing enough temporal context for the linear classifier to identify emotional onset, sustained response, and possible habituation patterns — all of which are absent from shorter windows.

### 4.5 Standard deviation analysis: a stable structural limitation

Across all configurations, fold-to-fold standard deviations are remarkably tight:

- Binary accuracy: 1.47%–2.20%
- 9-class accuracy: 2.05%–3.10%
- Binary AUROC: 0.019–0.026
- 9-class AUROC: 0.024–0.025

This consistency across ten random subject partitions confirms that the observed performance levels are **stable properties of the representation**, not artefacts of which subjects happen to fall in the test fold. The ceiling is structural: it reflects the degree of emotion–subject entanglement in REVE's frozen embedding space, which is constant across all folds.

---

## 5. Summary and Floor Metrics

| Task | Best config | Acc | AUROC | F1 | Chance | Gain |
| ------ | ------------ | ----- | ------- | ----- | -------- | ------ |

| Binary | nopool_flat, 10s/10s | **66.11%** | **0.6967** | **0.648** | 50.00% | +16.11 pp |
| 9-class | nopool_flat, 10s/10s | **38.02%** | **0.7568** | **0.374** | 11.11% | +26.91 pp |

### What this establishes

1. **REVE's frozen encoder carries real, transferable emotion signal.** A 9-class AUROC of 0.757 from a model that never saw THU-EP during pre-training — and whose weights are entirely frozen — is strong evidence for RQ1: large-scale EEG foundation models learn features that generalise to emotional categories in a zero-calibration, cross-subject setting. The 3.4× above-chance accuracy (38.02% vs. 11.11%) reinforces this conclusion.

2. **But the signal is entangled with subject identity.** The 52–55 pp train–val gap in the 9-class regime proves that REVE's features are heavily subject-specific. Within any individual's data, emotions are well-separated (90%+ train accuracy); across individuals, the decision boundaries learned from training subjects do not transfer. This is the "subject-variability wall" that motivates RQ2.

3. **The representation strategy matters critically.** Attention pooling (512-D) loses most of the emotion signal; the full spatial-temporal representation (168,960-D) retains it but at the cost of extreme dimensionality and overfitting. Neither a 512-D summary nor a 168,960-D unstructured vector is the right abstraction — what is needed is a *learned, subject-invariant compression* that preserves emotion-relevant features while discarding subject-specific ones. This is precisely what contrastive fine-tuning aims to achieve.

4. **Non-overlapping 10s windows provide the best signal-to-noise ratio.** Overlapping windows inflate training counts without adding genuine diversity, counterproductively accelerating overfitting. This informs the sampling strategy for all subsequent methods.

### Metrics to beat

Any method that fine-tunes the encoder or restructures the embedding space should exceed:

- **Binary:** 66.11% Acc / 0.6967 AUROC / 0.648 F1
- **9-class:** 38.02% Acc / 0.7568 AUROC / 0.374 F1
