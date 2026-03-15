# Baseline 1: Linear Probing on Frozen REVE Embeddings

## 1. 10-Fold Cross-Subject Cross-Validation

### 1.0 Protocol

**Evaluation:** 10-fold cross-subject cross-validation over 79 subjects (subject 75 excluded due to data corruption; stimuli-level exclusions for subjects 37 and 46). In each fold, approximately 71 subjects are used for training and 8 for validation. No subject appears in both the training and validation set of any fold. Fold assignments are deterministic (`FOLD_RANDOM_STATE = 42`).

**Architecture:** The frozen REVE encoder processes each EEG window and outputs a 4-D tensor of shape `(B, 30, H, 512)` — 30 EEG channels, H time patches (determined by window length), and 512 hidden dimensions. This tensor is then converted to a fixed-length vector using one of three representation strategies:

| Strategy | Notation | How | Dim (10 s window) |
| --- | --- | --- | --- |
| **Attention pooling** | `pool` | REVE's learnable `cls_query_token` aggregates all channels and time patches → `(B, 512)` | 512 |
| **No-pool, channel-mean** | `nopool_mean` | Mean over the 30-channel axis → `(B, H, 512)` → flatten | 5,632 |
| **No-pool, flatten** | `nopool_flat` | Flatten entire 4-D output → `(B, 30 × H × 512)` | 168,960 |

A single linear layer maps this vector to class logits. Training uses Adam (lr = 1e-3), batch size 64, max 80 epochs, and early stopping on validation accuracy with patience 15.

**Window configurations tested:**

| Label | Window | Stride | Overlap | Windows per 30 s stimulus |
| --- | --- | --- | --- | --- |
| w5s2 | 5 s | 2 s | 60 % | 13 |
| w5s5 | 5 s | 5 s | 0 % | 6 |
| w8s4 | 8 s | 4 s | 50 % | 6 |
| w10s5 | 10 s | 5 s | 50 % | 5 |
| w10s10 | 10 s | 10 s | 0 % | 3 |
| w30s4 | 30 s | 4 s | — | 1 (full stimulus) |

Note on **w30s4**: Since each stimulus lasts exactly 30 s (6,000 samples at 200 Hz), a 30 s window covers the entire stimulus, yielding exactly 1 window per stimulus regardless of stride. The stride parameter is therefore irrelevant for this configuration. This is a special case that is qualitatively different from the shorter-window experiments — it operates at the stimulus level rather than the sub-segment level.

---

### 1.1 Binary Task (Positive vs. Negative)

Random baseline: **50.00 %** (balanced classes after dropping the 4 neutral stimuli, leaving 12 negative + 12 positive = 24 stimuli).

#### 1.1.1 Effect of representation strategy (fixed 10 s window)

| Representation | Stride | Dim | Acc (mean ± std) | AUROC (mean ± std) | F1 (mean ± std) |
| --- | --- | --- | --- | --- | --- |
| pool | 5 s | 512 | 55.76 % ± 1.67 % | 0.562 ± 0.026 | 0.517 ± 0.072 |
| nopool_mean | 5 s | 5,632 | 59.88 % ± 1.47 % | 0.623 ± 0.019 | 0.603 ± 0.035 |
| nopool_flat | 5 s | 168,960 | 62.67 % ± 1.71 % | 0.648 ± 0.020 | 0.608 ± 0.054 |
| nopool_flat | 10 s | 168,960 | **66.11 % ± 1.65 %** | **0.697 ± 0.022** | **0.648 ± 0.041** |

**Interpretation.** The gap between pool (55.76 %) and nopool_flat (66.11 %) is 10.35 percentage points — far exceeding the ~1.7 % fold-level standard deviation — confirming that this is a structural effect of the representation rather than statistical noise.

REVE's `cls_query_token` was pre-trained via masked-patch reconstruction, an objective that incentivises capturing spatial layout and local waveform morphology for the purpose of reconstructing missing patches. Emotional state, however, is encoded in cross-channel coherence patterns, hemispheric asymmetries, and slow temporal dynamics — features that the reconstruction objective has no explicit incentive to preserve in the 512-D summary. When the linear classifier has direct access to the full 30 channels × H patches × 512 dimensions, it can locate and exploit these spatial-temporal emotion signatures that attention pooling compresses away.

The nopool_mean strategy (59.88 %) falls between pool and flat, consistent with the hypothesis that averaging across channels discards some but not all spatial information. It preserves temporal structure but collapses the channel dimension, losing hemispheric and topographic patterns that differentiate emotional states.

This finding has a direct implication: the question "can REVE overcome inter-subject variability?" is representation-dependent. At the pooled level, REVE barely exceeds chance on binary valence. At the unpooled level, it reaches nearly 70 % AUROC — evidence that the frozen encoder genuinely captures emotion-relevant EEG structure, but this structure is largely inaccessible through the standard pooling interface.

#### 1.1.2 Effect of window size and overlap (flat representation)

| Window | Stride | Overlap | Dim | Windows / stim | Acc (mean ± std) | AUROC (mean ± std) | F1 (mean ± std) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 s | 2 s | 60 % | 76,800 | 13 | 57.54 % ± 1.70 % | 0.596 ± 0.026 | 0.551 ± 0.054 |
| 5 s | 5 s | 0 % | 76,800 | 6 | 60.08 % ± 1.66 % | 0.625 ± 0.019 | 0.611 ± 0.022 |
| 8 s | 4 s | 50 % | 122,880 | 6 | 61.07 % ± 2.20 % | 0.638 ± 0.024 | 0.587 ± 0.068 |
| 10 s | 5 s | 50 % | 168,960 | 5 | 62.67 % ± 1.71 % | 0.648 ± 0.020 | 0.608 ± 0.054 |
| 10 s | 10 s | 0 % | 168,960 | 3 | 66.11 % ± 1.65 % | 0.697 ± 0.022 | 0.648 ± 0.041 |
| **30 s** | **4 s** | **—** | **~460,800** | **1** | **82.22 % ± 1.73 %** | **0.847 ± 0.020** | **0.827 ± 0.020** |

Two clear trends emerge from this table:

**Trend 1: Longer windows improve performance.** Accuracy increases monotonically with window length — from 57.54 % (5 s) to 66.11 % (10 s) to 82.22 % (30 s). At 30 s, each window covers the full stimulus, giving the classifier access to the entire temporal trajectory of the emotional response: onset, sustained response, and any habituation or recovery dynamics. Shorter windows capture only a fragment of this trajectory, and the classifier must infer emotional state from partial temporal evidence. The jump from 10 s to 30 s (+16.11 pp) is particularly striking, and Section 2 will investigate whether it reflects genuine emotion recognition or confounds.

**Trend 2: Non-overlapping windows outperform overlapping ones.** Comparing configurations that differ only in overlap:

| Pair | Overlapping | Non-overlapping | Δ Acc | Δ AUROC |
| --- | --- | --- | --- | --- |
| 5 s window | w5s2 (60 %): 57.54 % | w5s5 (0 %): 60.08 % | **+2.54 pp** | **+0.030** |
| 10 s window | w10s5 (50 %): 62.67 % | w10s10 (0 %): 66.11 % | **+3.44 pp** | **+0.049** |

In both cases, the non-overlapping configuration achieves higher validation accuracy despite having fewer training samples (roughly half as many windows). Overlapping windows share substantial temporal content — consecutive windows with 50 % overlap contain identical EEG segments in shifted context — creating correlated pseudo-replication. This inflates the apparent training set size without adding genuine informational diversity. The linear classifier exploits these intra-stimulus correlations to achieve high training accuracy, but the correlations are stimulus-specific and do not generalise to unseen subjects. Non-overlapping windows provide temporally independent segments, producing a more honest training signal.

#### 1.1.3 Overfitting analysis

The following table is reconstructed from the training logs at the epoch selected by early stopping:

| Config | Dim | Val Acc | Approx. linear params |
| --- | --- | --- | --- |
| pool w10s5 | 512 | 55.76 % | ~1,026 |
| nopool_mean w10s5 | 5,632 | 59.88 % | ~11,266 |
| nopool_flat w10s5 | 168,960 | 62.67 % | ~337,922 |
| nopool_flat w10s10 | 168,960 | 66.11 % | ~337,922 |
| nopool_flat w30s4 | ~460,800 | 82.22 % | ~921,602 |

The pool configuration with ~1,000 parameters and ~8,500 training windows is in a low-capacity regime where the linear head simply cannot overfit — but this is not a sign of good generalisation; it is a sign that the representation contains too little information for even a minimal classifier to learn from. The flat configurations are in the opposite regime: with ~338K–922K parameters and ~1,700–5,100 training windows, the model is heavily over-parameterised. Yet it achieves the best validation metrics, confirming that the additional capacity grants access to genuine cross-subject emotion structure — alongside the inevitable memorisation of subject-specific patterns. The gap between training accuracy (routinely 85–95 %) and validation accuracy represents the "subject identity tax": the portion of classifier capacity consumed by subject-specific confounders rather than universal emotion features.

---

### 1.2 Nine-Class Task (9 Emotions)

Random baseline: **11.11 %** (1/9). The nine classes are: Anger, Disgust, Fear, Sadness (negative valence); Neutral; Amusement, Inspiration, Joy, Tenderness (positive valence). Each emotion has 3 stimuli except Neutral which has 4, totalling 28 stimuli.

| Config | Window | Stride | Overlap | Acc (mean ± std) | AUROC (mean ± std) | F1 (mean ± std) |
| --- | --- | --- | --- | --- | --- | --- |
| pool | 10 s | 10 s | 0 % | 22.21 % ± 1.67 % | 0.631 ± 0.023 | 0.209 ± 0.017 |
| nopool_flat | 10 s | 5 s | 50 % | 34.14 % ± 3.10 % | 0.706 ± 0.025 | 0.334 ± 0.033 |
| nopool_flat | 10 s | 10 s | 0 % | 38.02 % ± 2.05 % | 0.757 ± 0.024 | 0.374 ± 0.025 |
| nopool_flat | **30 s** | **4 s** | **—** | **71.66 % ± 5.11 %** | **0.898 ± 0.015** | **0.720 ± 0.050** |

**Interpretation.**

The pool (512-D) configuration achieves 22.21 % accuracy and 0.631 AUROC, already a substantial gap above chance (11.11 %). However, the gap between pool (22.21 %) and nopool_flat w10s10 (38.02 %) is +15.81 pp — consistent with the binary finding that bypassing pooling unlocks access to spatial-temporal emotion structure that the compressed summary discards. Notably, the AUROC gap is also pronounced (0.631 vs 0.757, Δ = 0.126), indicating that pool not only makes worse argmax decisions but also produces less well-ordered probability rankings across the 9 classes.

The same two trends observed in binary (longer windows and non-overlapping strides help) hold for 9-class, but the magnitudes are more dramatic:

- **Non-overlapping vs overlapping (10 s):** +3.88 pp accuracy, +0.051 AUROC. The improvement from eliminating overlap is proportionally similar to binary.
- **10 s → 30 s:** +33.64 pp accuracy (38.02 % → 71.66 %), +0.141 AUROC. This tripling of performance is far more than what window length alone should provide if the classifier were recognising emotions from neural dynamics.

The 30 s result (71.66 %, 0.898 AUROC) is extraordinarily high for cross-subject 9-class EEG emotion recognition — comparable to or exceeding state-of-the-art results in the literature that use far more sophisticated models and explicitely address the inter-subject variability problem. This warrants serious scrutiny (addressed in Section 2).

**Standard deviation analysis.** The fold-level std for w30s4 9-class is 5.11 % — substantially higher than the 2–3 % observed in shorter-window configurations. This increased variance arises because each fold's validation set contains only ~8 subjects × 27 stimuli × 1 window = ~216 samples, compared to ~8 × 27 × 3 = ~648 samples for w10s10. With fewer samples, individual fold estimates become noisier. Importantly, even at the lower end of fold performance (64.89 % for fold 6), accuracy remains 5.8× above chance, confirming that the phenomenon is robust.

**AUROC vs accuracy dissociation.** A noteworthy pattern across all 9-class configurations: AUROC consistently exceeds what accuracy alone might suggest. For w10s10, 38.02 % accuracy coexists with 0.757 AUROC. This dissociation indicates that when the classifier is wrong, it typically assigns high probability to a semantically related class (e.g., confusing Joy with Amusement, or Anger with Disgust), preserving the probability ranking even when the argmax prediction is incorrect. The model captures the structure of the emotional space even if it cannot always resolve fine-grained distinctions across subjects.

---

## 2. Stimulus-Generalization Evaluation

### 2.0 Motivation

The 30 s flat results (82.22 % binary, 71.66 % 9-class) are suspiciously high. When a window covers the entire stimulus, the linear classifier has access to a temporal fingerprint that uniquely identifies which of the 28 stimuli was presented. The classifier may learn "stimulus X → neural pattern Y → emotion Z" rather than "neural pattern X → emotion Y", achieving high accuracy through stimulus identification rather than emotion recognition.

This is not a bug in the evaluation protocol — it is a legitimate threat to validity. Standard cross-subject cross-validation does not control for it because the same stimuli appear in both training and validation sets (just from different subjects). If stimulus-specific temporal fingerprints are shared across subjects (which they are, since all subjects watched the same videos), the classifier can exploit them.

### 2.1 Protocol

To disentangle emotion recognition from stimulus fingerprinting, we introduce a **cross-subject × cross-stimulus generalisation** evaluation. The design stacks two independent sources of held-out variation:

1. **Subject split** — identical to standard CV: ~71 subjects for training, ~8 for validation per fold.
2. **Stimulus split** — from the training subjects, only 2/3 of the stimuli per emotion are used for training; from the validation subjects, only the complementary held-out 1/3 of stimuli are used for evaluation.

Concretely: `train set = train_subjects × train_stimuli`, `val set = val_subjects × held_out_stimuli`. The held-out stimuli are entirely absent from training — neither the subjects watching them nor the stimuli themselves are seen during training. Any accuracy above chance must therefore derive from emotion-related neural patterns that generalise across both individuals and stimulus exemplars.

**Stimulus split procedure.** For each of the 9 emotion categories (Anger, Disgust, Fear, Sadness, Neutral, Amusement, Inspiration, Joy, Tenderness), the stimuli are divided approximately 2/3 for training and 1/3 for testing. For the 8 categories with 3 stimuli each, this yields 2 train / 1 test; for Neutral (4 stimuli), this yields 3 train / 1 test. The split is performed independently per emotion group to ensure that every emotion is represented in both partitions. A random split across all stimuli could accidentally exclude all exemplars of an emotion from the test set, making the evaluation uninformative for that category.

In binary mode, Neutral stimuli are excluded (as in standard CV), and the per-emotion structure is maintained even though the classifier only sees positive/negative labels. This ensures that the held-out stimuli span the full emotional spectrum within each valence class.

**Multi-seed robustness.** Because most emotions have only 3 stimuli, there are exactly 3 possible held-out choices per emotion group (4 for Neutral). To ensure that conclusions do not depend on a particular split, the evaluation is repeated with 5 different random seeds (123, 456, 789, 101, 202). Each seed produces a different stimulus partition; the 10-fold cross-subject split remains identical across seeds. The reported metrics are averages across seeds, with standard deviations quantifying sensitivity to split choice.

**Resulting sample sizes.** In binary mode (8 emotions × 3 stimuli, Neutral dropped): 16 train stimuli, 8 test stimuli. In 9-class mode (8 emotions × 3 stimuli + 4 Neutral): 19 train stimuli, 9 test stimuli. Combined with ~71/~8 subject split, the validation set contains windows from subjects who were never in training, watching stimuli that were never in training.

### 2.2 Binary Task — Generalization Results

| Config | Std CV Acc | Std CV AUROC | Gen Acc (5-seed) | Gen AUROC (5-seed) | Δ Acc | Δ AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| pool w10s10 | ~55.76 %† | ~0.562† | 54.57 % ± 1.16 % | 0.523 ± 0.020 | ~−1.19 pp | ~−0.039 |
| nopool_flat w10s10 | 66.11 % ± 1.65 % | 0.697 ± 0.022 | 57.29 % ± 0.93 % | 0.569 ± 0.011 | **−8.82 pp** | **−0.128** |
| nopool_flat w30s4 | 82.22 % ± 1.73 % | 0.847 ± 0.020 | 60.78 % ± 2.33 % | 0.587 ± 0.031 | **−21.44 pp** | **−0.260** |

† Standard CV was run for pool at w10s5 (55.76 %, 0.562 AUROC). No standard CV run exists for pool at w10s10; the pool representation aggregates the entire window into a fixed 512-D vector regardless of window length, so performance is expected to differ only marginally.

**Per-seed breakdown (accuracy only):**

| Seed | pool w10s10 gen | w10s10 gen | w30s4 gen |
| --- | --- | --- | --- |
| 123 | 53.89 % | 58.59 % | 64.58 % |
| 456 | 52.93 % | 56.92 % | 59.66 % |
| 789 | 54.81 % | 57.81 % | 60.56 % |
| 101 | 55.44 % | 56.17 % | 58.35 % |
| 202 | 55.77 % | 56.98 % | 60.73 % |
| **Mean ± std** | **54.57 % ± 1.16 %** | **57.29 % ± 0.93 %** | **60.78 % ± 2.33 %** |

**Interpretation.**

The 30 s configuration suffers the most dramatic collapse: from 82.22 % to 60.78 % (−21.44 pp). This confirms that a substantial portion of the 30 s performance is attributable to stimulus fingerprinting. When the classifier cannot rely on recognising familiar stimuli, it retains only the portion of its decision boundary that captures genuine emotion-related neural patterns. The residual accuracy (60.78 %) is still above chance (50 %) but only marginally above the 10 s generalization result (57.29 %), suggesting that the true emotion-recognition signal in the frozen REVE representation is comparable across window lengths.

The 10 s nopool_flat configuration drops by −8.82 pp, a smaller but still meaningful decline. Even at 10 s (covering one-third of the stimulus), there is some degree of stimulus-specific temporal structure that the classifier exploits. However, the relative stability of the 10 s result suggests that shorter windows derive more of their accuracy from genuine neural patterns.

The pool representation exhibits a strikingly different behaviour: it drops by only ~1.19 pp (from ~55.76 % to 54.57 %). This near-immunity to stimulus removal, however, should not be interpreted as evidence of superior generalisation. Rather, it reflects the fact that pool barely learned anything from either source — genuine emotion or stimulus fingerprints — during standard CV. The low standard CV performance (55.76 %) leaves almost no room for a generalization penalty: there is no elevated performance to collapse. This creates a paradoxical inversion where the worst-performing representation suffers the smallest absolute degradation under the generalization challenge. The appropriate interpretation is that pool lies at the performance floor and stimulus removal simply exposes how little signal either pathway was contributing.

The cross-seed standard deviations confirm stability across all three configurations: ±1.16 % for pool, ±0.93 % for nopool_flat w10s10, and ±2.33 % for nopool_flat w30s4. The higher variability for w30s4 reflects the smaller effective validation set (1 window per stimulus rather than 3–5), making individual fold estimates more sensitive to which particular stimuli are held out.

### 2.3 Nine-Class Task — Generalization Results

| Config | Std CV Acc | Std CV AUROC | Gen Acc (5-seed) | Gen AUROC (5-seed) | Δ Acc | Δ AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| pool w10s10 | 22.21 % ± 1.67 % | 0.631 ± 0.023 | 17.88 % ± 1.06 % | 0.556 ± 0.011 | **−4.33 pp** | **−0.075** |
| nopool_flat w10s5 | 34.14 % ± 3.10 % | 0.706 ± 0.025 | 17.73 % ± 0.64 % | 0.560 ± 0.005 | **−16.41 pp** | **−0.146** |
| nopool_flat w10s10 | 38.02 % ± 2.05 % | 0.757 ± 0.024 | 18.84 % ± 0.64 % | 0.566 ± 0.005 | **−19.18 pp** | **−0.191** |
| nopool_flat w30s4 | 71.66 % ± 5.11 % | 0.898 ± 0.015 | 20.49 % ± 1.52 % | 0.554 ± 0.012 | **−51.17 pp** | **−0.344** |

**Per-seed breakdown (accuracy only):**

| Seed | pool w10s10 gen | nopool_flat w10s5 gen | nopool_flat w10s10 gen | nopool_flat w30s4 gen |
| --- | --- | --- | --- | --- |
| 123 | 17.66 % | 17.09 % | 19.07 % | 22.26 % |
| 456 | 18.28 % | 18.77 % | 19.27 % | 19.42 % |
| 789 | 16.29 % | 17.67 % | 17.72 % | 21.55 % |
| 101 | 19.20 % | 17.37 % | 19.19 % | 20.69 % |
| 202 | 17.98 % | 17.75 % | 18.94 % | 18.55 % |
| **Mean ± std** | **17.88 % ± 1.06 %** | **17.73 % ± 0.64 %** | **18.84 % ± 0.64 %** | **20.49 % ± 1.52 %** |

**Interpretation.**

The 9-class generalization results are the most informative in the entire experiment set, as they reveal the effective ceiling of the frozen REVE representation for cross-subject, cross-stimulus emotion recognition.

The 30 s configuration collapses from 71.66 % to 20.49 % — a drop of 51.17 pp that erases almost all of the 60.55 pp advantage over chance (71.66 % − 11.11 % = 60.55 pp; of this, only 20.49 % − 11.11 % = 9.38 pp survives). Stated differently, **84.5 % of the above-chance performance in the 30 s configuration was attributable to stimulus fingerprinting.** The remaining 9.38 pp above chance, while statistically non-zero, represents a weak signal — the model has barely learned to differentiate 9 emotions when it cannot rely on recognising familiar stimuli.

The pool configuration drops from 22.21 % to 17.88 % (−4.33 pp). This is a smaller absolute drop than the flat configurations, but the same asymmetry noted in binary applies: pool had less standard CV performance to lose. Relative to its above-chance margin (22.21 % − 11.11 % = 11.10 pp above chance), pool loses 4.33 pp — roughly 39 % of its above-chance margin. By comparison, nopool_flat w10s10 loses 19.18 pp out of 26.91 pp (71 % of its margin) and nopool_flat w10s5 loses 16.41 pp out of 23.03 pp (71 % of its margin). The proportional loss is therefore consistent across flat configurations and only appears smaller for pool because the absolute margin was lower. This uniformity suggests that approximately 70 % of the above-chance performance in any 10 s flat configuration is attributable to stimulus-correlated EEG patterns rather than genuine emotion-category-level generalisation.

**Universal convergence under generalization.** The most significant observation is the convergence of all four configurations to a narrow band of 17–21 % under stimulus generalization, despite spanning a 50 pp range under standard CV (22.21 %–71.66 %). This convergence is not incidental — it is a direct consequence of removing stimulus identity as a shared cue. Once the classifier must generalise to entirely novel stimuli, all representations — whether pooled, flat, 10 s, or 30 s — access approximately the same residual emotion signal. This signal amounts to roughly 7–9 pp above chance (a relative lift of 63–81 % above the 11.11 % baseline), consistent across all seeds and all window configurations.

This convergence strongly suggests the existence of a **representation-imposed ceiling** on cross-subject cross-stimulus emotion recognition with the frozen REVE encoder: no matter how the 4-D output tensor is exposed to the linear classifier (pooled, flattened, at 5 s, 10 s, or 30 s), the amount of truly transferable emotion information is approximately constant at ~7–9 pp above random. The ceiling is not a property of the linear probe design but of the frozen encoder itself — the information simply is not present in a form that a linear transformation can extract stably across both subject and stimulus variation simultaneously.

The AUROC collapse is particularly revealing. Under standard CV, the 30 s model achieved 0.898 AUROC — near-perfect ranking. Under generalization, it falls to 0.554, barely above the 0.500 random baseline. All four generalization AUROCs cluster tightly between 0.554 and 0.566 — a range so narrow (0.012 wide) that it confirms the configurations have reached a common floor. The classifier has lost not just calibration but its ability to rank classes correctly. This eliminates the alternative explanation that the model learned the correct emotional structure but merely became less confident — it has lost the structure entirely.

**Cross-seed stability.** The per-seed standard deviations across all four configurations are small: ±1.06 % (pool), ±0.64 % (nopool_flat w10s5), ±0.64 % (nopool_flat w10s10), ±1.52 % (nopool_flat w30s4). The stability confirms that the convergence result is not artefactual — it persists regardless of which specific stimuli are held out for evaluation. The slightly higher variability for w30s4 reflects the 1-window-per-stimulus constraint in the validation set, which amplifies sensitivity to individual stimulus choice.

---

## 3. Summary

### 3.1 Key findings

1. **REVE's attention pooling is a severe bottleneck for emotion recognition.** The pooled 512-D representation (55.76 % binary, 22.21 % 9-class at w10s10) discards most emotion-relevant information. Bypassing pooling and exposing the full spatial-temporal tensor to the classifier recovers +10.35 pp binary and +15.81 pp 9-class (at w10s10). The pre-training objective (masked-patch reconstruction) optimises for local waveform fidelity, not the cross-channel and cross-temporal patterns that encode emotional state.

2. **Non-overlapping windows consistently outperform overlapping ones.** Despite producing fewer training samples, stride-equals-window configurations improve generalisation at every window length tested (+2.54 pp at 5 s, +3.44 pp at 10 s). Overlapping windows create correlated pseudo-replication that inflates training accuracy without adding genuine informational diversity. **Non-overlapping windows should be the default for all subsequent experiments.**

3. **The 30 s flat configuration is dominated by stimulus fingerprinting.** Standard CV accuracy (82.22 % binary, 71.66 % 9-class) collapses under cross-stimulus generalization to 60.78 % and 20.49 %, respectively. For 9-class, 84.5 % of the above-chance performance is attributable to stimulus identity rather than emotion recognition. This finding demonstrates that **standard cross-subject cross-validation on shared-stimulus designs is insufficient for validating emotion recognition** — stimulus-generalization controls are essential.

4. **Under generalization, performance converges across all configurations.** The large gaps between configurations under standard CV nearly vanish under generalization — all four tested configurations (pool w10s10, nopool_flat w10s5, nopool_flat w10s10, nopool_flat w30s4) converge to 17–21 % 9-class accuracy and 54–61 % binary accuracy. The "extra" information exploited by longer windows and richer representations under standard CV is almost entirely stimulus-specific rather than genuinely emotion-related.

5. **Approximately 70 % of above-chance performance in flat 10 s configurations is attributable to stimulus-correlated structure.** Across nopool_flat w10s5 and w10s10, the relative loss of above-chance margin under generalization is approximately 71 % in both cases. This quantifies the stimulus fingerprinting contribution even at short windows, and motivates the use of the generalization protocol as the primary metric for future evaluations.

6. **The frozen REVE representation carries a real but weak cross-subject, cross-stimulus emotion signal.** Under the most rigorous evaluation, binary accuracy is ~55–61 % (5–11 pp above chance) and 9-class accuracy is ~18–20 % (7–9 pp above chance). This signal is consistently above chance across all 5 seeds and all 10 folds, confirming that it is real. However, it is far too weak to be practically useful, and it is effectively constant regardless of representation strategy or window length. This establishes the need for methods that actively reorganise the embedding space to amplify emotion-relevant structure and suppress subject-specific and stimulus-specific confounders.

### 3.2 Consolidated results table

| Task | Config | Standard CV Acc | Standard CV AUROC | Gen Acc (5-seed) | Gen AUROC (5-seed) |
| --- | --- | --- | --- | --- | --- |
| Binary | pool w10s5 | 55.76 % ± 1.67 % | 0.562 ± 0.026 | — | — |
| Binary | nopool_mean w10s5 | 59.88 % ± 1.47 % | 0.623 ± 0.019 | — | — |
| Binary | nopool_flat w5s2 | 57.54 % ± 1.70 % | 0.596 ± 0.026 | — | — |
| Binary | nopool_flat w5s5 | 60.08 % ± 1.66 % | 0.625 ± 0.019 | — | — |
| Binary | nopool_flat w8s4 | 61.07 % ± 2.20 % | 0.638 ± 0.024 | — | — |
| Binary | nopool_flat w10s5 | 62.67 % ± 1.71 % | 0.648 ± 0.020 | — | — |
| Binary | nopool_flat w10s10 | 66.11 % ± 1.65 % | 0.697 ± 0.022 | 57.29 % ± 0.93 % | 0.569 ± 0.011 |
| Binary | nopool_flat w30s4 | 82.22 % ± 1.73 % | 0.847 ± 0.020 | 60.78 % ± 2.33 % | 0.587 ± 0.031 |
| Binary | pool w10s10 | — | — | 54.57 % ± 1.16 % | 0.523 ± 0.020 |
| 9-class | pool w10s10 | 22.21 % ± 1.67 % | 0.631 ± 0.023 | 17.88 % ± 1.06 % | 0.556 ± 0.011 |
| 9-class | nopool_flat w10s5 | 34.14 % ± 3.10 % | 0.706 ± 0.025 | 17.73 % ± 0.64 % | 0.560 ± 0.005 |
| 9-class | nopool_flat w10s10 | 38.02 % ± 2.05 % | 0.757 ± 0.024 | 18.84 % ± 0.64 % | 0.566 ± 0.005 |
| 9-class | nopool_flat w30s4 | 71.66 % ± 5.11 % | 0.898 ± 0.015 | 20.49 % ± 1.52 % | 0.554 ± 0.012 |

### 3.3 Baseline metrics to beat

Any method that fine-tunes the encoder or restructures the embedding space must be evaluated under **both** standard CV and cross-stimulus generalization. The relevant baselines are:

**Standard cross-subject CV (nopool_flat w10s10):**

- Binary: 66.11 % Acc / 0.697 AUROC / 0.648 F1
- 9-class: 38.02 % Acc / 0.757 AUROC / 0.374 F1

**Cross-stimulus generalization (nopool_flat w10s10, 5-seed average):**

- Binary: 57.29 % Acc / 0.569 AUROC
- 9-class: 18.84 % Acc / 0.566 AUROC

The generalization metrics are arguably more important than the standard CV metrics, as they measure whether the model has learned transferable emotion representations rather than stimulus-specific patterns. A method that improves standard CV accuracy without improving generalization accuracy has merely learned to fingerprint stimuli more effectively.
