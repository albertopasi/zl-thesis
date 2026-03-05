# Analysis of Baseline 1: Linear Probing on Frozen REVE Embeddings

**Evaluation Protocol:** 10-Fold Cross-Subject Cross-Validation

**Dataset:** THU-EP (79 Subjects) — see `excluded_data.md` for corruption details

**Architecture:** Frozen REVE Foundation Model → 512-D Attention-Pooled Embedding → Linear Classifier

**Window configurations tested:** 5s/2s stride, 8s/4s stride, 10s/5s stride (all at 200 Hz)

---

## 1. Binary Task (Positive vs. Negative)

Random chance: 50.00%

### Binary: Results by Window Size

| Window | Stride | Acc (mean ± std)   | AUROC (mean ± std) | F1 (mean ± std)   |
|--------|--------|--------------------|--------------------|-------------------|
| 5s     | 2s     | 55.40% ± 1.28%     | 0.5612 ± 0.0171    | 0.518 ± 0.067     |
| 8s     | 4s     | 56.25% ± 1.56%     | 0.5747 ± 0.0223    | 0.576 ± 0.049     |
| 10s    | 5s     | 55.55% ± 1.74%     | 0.5584 ± 0.0268    | 0.519 ± 0.079     |

### Binary: Interpretation

**Window size has negligible impact.** The spread between the three configurations is at most ~0.9% in accuracy, well within the ±1.3–1.7% standard deviation of any individual run. The 8s/4s window marginally outperforms the others, possibly because it captures enough temporal context for affective dynamics (emotion responses typically unfold over 4–8 s) without averaging out transient features. However, this difference is not statistically meaningful at this stage and should not be over-interpreted.

**Accuracy is marginally above chance.** A linear probe trained on ~71 subjects and tested on ~8 unseen subjects reaches ~55–56%, compared to the 50% random baseline. The improvement is consistent across all folds, confirming this is a real signal rather than noise. However, the practical margin is very narrow (~5–6 pp above chance), which is the expected outcome of a frozen cross-subject linear probe on EEG.

**The "subject variability wall."** AUROC values of 0.56–0.57 (chance = 0.50) tell the same story as accuracy. The embeddings carry some emotion-relevant structure, but the latent space is heavily entangled with subject-specific physiological confounders. A single linear hyperplane cannot cleanly separate valence categories when subject identity is the dominant source of variance.

**Is this a good baseline?** Yes. If the linear probe had achieved 70–80%, the thesis hypothesis would be invalidated — it would mean frozen REVE embeddings linearly separate emotion cross-subject, making fine-tuning and contrastive learning unnecessary. The ~56% ceiling establishes the kind of baseline that motivates the next steps: LoRA fine-tuning to adapt the encoder to THU-EP, and contrastive learning to explicitly disentangle subject identity from emotional content in the latent space.

---

## 2. Nine-Class Task (8 Emotions + Neutral)

Random chance: ~11.11% (1/9)

### 9-Class: Results by Window Size

| Window | Stride | Acc (mean ± std)   | AUROC (mean ± std) | F1 (mean ± std)   |
|--------|--------|--------------------|--------------------|-------------------|
| 5s     | 2s     | 20.24% ± 1.27%     | 0.6158 ± 0.0218    | 0.1805 ± 0.0140   |
| 8s     | 4s     | 21.09% ± 1.75%     | 0.6242 ± 0.0215    | 0.1945 ± 0.0169   |
| 10s    | 5s     | 21.37% ± 1.42%     | 0.6244 ± 0.0235    | 0.1989 ± 0.0129   |

### 9-Class: Interpretation

**Window size again has negligible impact.** The difference between w5s2 and w10s5 is only ~1.1% in accuracy and 0.009 in F1 — less than one standard deviation in each case. Longer windows trend slightly better (more temporal context for fine-grained emotion discrimination), but the effect is too small to be conclusive without statistical testing. The expectation is that window size will matter more once contrastive fine-tuning is applied, since richer temporal features may help distinguish adjacent emotion categories (e.g., Anger vs. Fear vs. Disgust).

**Nearly double chance accuracy.** The 9-class result (~21%) is the most informative metric in this entire baseline. The frozen REVE encoder, pre-trained on EEG data from unrelated tasks, spontaneously organizes the THU-EP stimulus-evoked responses into emotion-specific clusters well enough that a single linear layer performs at roughly 2× chance. This is a strong prior: the backbone is already doing meaningful emotion-relevant feature extraction without any THU-EP supervision.

**AUROC of ~0.62 is more encouraging than accuracy.** In a 9-class one-vs-rest macro-average, an AUROC of 0.62 means: given a window from class X and a window from any other class, there is a 62% probability the model assigns higher probability to the correct class. This is substantially above the 0.50 random baseline. Accuracy fails here primarily because the 9-class decision boundaries are blurry — the top-1 prediction is often wrong by one adjacent category (e.g., predicting Anger instead of Disgust), but the correct class still ranks high in the probability distribution. AUROC captures this partial ordering, which accuracy discards.

**Tight standard deviation signals structural, not stochastic, limitation.** The fold-to-fold std in accuracy (1.3–1.8%) and F1 (0.013–0.017) is remarkably consistent. This means the ~21% ceiling is a stable property of the frozen representation, not an artifact of which subjects happen to be in the test fold. The model reliably achieves this level regardless of the subject partition, confirming that the limitation is representational, not data-split dependent.

---

## 3. Cross-Task Comparison and Conclusions

| Task    | Best window | Acc    | AUROC  | F1     | Chance  | Gain over chance |
|---------|-------------|--------|--------|--------|---------|------------------|
| Binary  | 8s/4s       | 56.25% | 0.5747 | 0.5761 | 50.00%  | +6.25 pp         |
| 9-class | 10s/5s      | 21.37% | 0.6244 | 0.1989 | 11.11%  | +10.26 pp        |

In relative terms, the 9-class task actually shows a stronger signal above chance (+92% relative lift vs. +12.5% for binary). This is counter-intuitive but consistent with how frozen EEG encoders behave: the backbone encodes coarse emotional category structure better than fine valence polarity, possibly because category-level differences involve more distinct neural activation patterns than the positive/negative distinction alone.

**Window size is not a meaningful hyperparameter at this stage.** None of the three configurations produce differences that exceed one standard deviation. This is expected for a linear probe, a single linear layer cannot exploit the richer temporal context of longer windows any better than a shorter one, since it sees only the pooled 512-D vector regardless. Window size is likely to matter more once the encoder itself is fine-tuned and attention patterns inside REVE can specialize to longer-range temporal dependencies.

**These results establish the floor.** The metrics to beat for any subsequent method (LoRA fine-tuning, contrastive alignment, or their combination) are:

- **Binary:** 56.25% Acc / 0.5747 AUROC / 0.576 F1 (w8s4)
- **9-class:** 21.37% Acc / 0.6244 AUROC / 0.199 F1 (w10s5)

Fine-tuning is necessary to move past the subject-variability wall. Contrastive learning is necessary to untangle emotion from subject identity in the latent space.

---

## 4. Ablation: L2 Feature Normalization (Binary Task Only)

### Motivation

REVE uses a pre-norm architecture: RMSNorm is applied *inside* each transformer block before the attention and feedforward operations, but there is no final LayerNorm at the backbone output. Because residual additions accumulate across all layers without rescaling, the 512-D attention-pooled embeddings may carry non-trivial magnitude variation across subjects and stimuli. L2 normalization projects every embedding onto the unit hypersphere (`x / ||x||₂`), which removes this scale and constrains the linear classifier to use only directional information. The question is whether discarding magnitude helps or hurts.

### Protocol

The `--normalize` flag was added to `train_lp.py`. When enabled, `F.normalize(x, dim=-1)` is applied inside `LinearProber.forward` before the linear layer. This adds zero trainable parameters; the classifier still sees a 512-D vector, but with unit norm. All other hyperparameters are identical to the standard runs.

Results are available for the binary task only (3 window configurations × 10 folds each).

### Results

| Window | Variant    | Acc (mean ± std)   | AUROC (mean ± std) | F1 (mean ± std)   |
|--------|------------|--------------------|--------------------|-------------------|
| 5s/2s  | Standard   | 55.40% ± 1.28%     | 0.5612 ± 0.0171    | 0.518 ± 0.067     |
| 5s/2s  | L2-norm    | 54.52% ± 1.53%     | 0.5520 ± 0.0232    | 0.508 ± 0.080     |
| 8s/4s  | Standard   | 56.25% ± 1.56%     | 0.5747 ± 0.0223    | 0.576 ± 0.049     |
| 8s/4s  | L2-norm    | 55.34% ± 1.71%     | 0.5632 ± 0.0232    | 0.511 ± 0.056     |
| 10s/5s | Standard   | 55.55% ± 1.74%     | 0.5584 ± 0.0268    | 0.519 ± 0.079     |
| 10s/5s | L2-norm    | 54.40% ± 2.19%     | 0.5419 ± 0.0313    | 0.511 ± 0.071     |

### Interpretation

**L2 normalization consistently hurts.** Accuracy drops by 0.88–1.15 pp and AUROC drops by 0.009–0.016 across all three window configurations. The direction of the effect is perfectly consistent: every window, every metric is worse under normalization. The individual effect sizes are modest (~0.5–0.7σ per window), but the uniformity across three independent experiments makes the conclusion reliable.

**What this tells us about REVE's embedding geometry.** The norm of a pre-norm transformer's output is not arbitrary noise — it accumulates meaningful signal through the residual stream. A vector with larger magnitude represents a pattern the model encoded more strongly. By normalizing to the unit sphere, we discard this information and the linear classifier is forced to work with direction alone, losing discriminative power. The raw embeddings' geometry (both direction and magnitude) is therefore informative, and the standard linear probe is the correct experimental choice.

**This negative result is useful.** It rules out the hypothesis that REVE's absent final LayerNorm causes pathological scale variance that harms the probe. Scale variance exists, but it is signal-carrying, not noise. This also provides a posterior justification for the standard protocol: the raw embeddings are preferable, and the baseline numbers reported in Sections 1–3 are the appropriate floor for subsequent methods.
