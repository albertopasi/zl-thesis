# Experimental Plan: LoRA Finetuning + Supervised Contrastive Learning

## Context

The linear probing baseline (Baseline 1) established that the frozen REVE encoder carries a real but weak cross-subject emotion signal — under the most rigorous evaluation (cross-subject × cross-stimulus generalization), all configurations converge to ~18–20% 9-class accuracy and ~55–61% binary accuracy, regardless of representation strategy or window length. This representation-imposed ceiling motivates two follow-up phases: (1) adapting the encoder via LoRA to reorganise the representation space, and (2) adding supervised contrastive learning to explicitly enforce subject-invariant, emotion-discriminative geometry.

---

## Scientific Soundness of the 3-Phase Plan

The progression frozen → LoRA-finetuned → LoRA+SupCon is a clean ablation ladder. Each phase isolates a distinct research question:

| Phase | Research question | What changes vs. previous |
|---|---|---|
| 1 (done) | How much emotion info does frozen REVE contain? | — (baseline) |
| 2 (next) | Does LoRA adaptation reorganise the space for emotion? | Encoder adapts; same head + loss |
| 3 (contribution) | Does SupCon's geometric inductive bias improve subject invariance? | Loss changes; same LoRA config |

**Phase 2 is the critical control.** Without it, one cannot attribute improvements in Phase 3 to SupCon vs. LoRA. Many papers skip this and overclaim their contrastive loss contribution.

**SupCon is well-motivated over alternatives:**

- SimCLR/MoCo require EEG augmentations (poorly validated, introduce confounds)
- Triplet loss requires explicit hard-pair mining and scales poorly
- SupCon uses label-defined positives, handles all pairwise comparisons naturally, and directly optimises for the needed property: same-emotion samples from different subjects nearby in embedding space

---

## Configuration Selection

### Primary: w10s10 (10s window, 10s stride, non-overlapping)

- Best generalization performance in LP (18.84% 9-class, 57.29% binary)
- Non-overlapping confirmed superior in LP: +2.5–3.4 pp over overlapping at every window length
- 3 windows/stimulus → ~5,100 binary train samples, ~5,960 for 9-class standard CV
- 11 temporal patches → manageable 4-D tensor (30 × 11 × 512)
- Sufficient for contrastive learning with batch_size ≥ 128

### Secondary (if compute allows): w5s5 (5s window, 5s stride)

- 6 windows/stimulus → ~10,200 binary train samples (2× more than w10s10)
- Favours SupCon's data hunger; smaller input tensor allows larger batches
- Provides "more data vs. more temporal context" comparison
- Lower LP gen performance, but the gap may close with finetuning

### Excluded

- **w5s2, w10s5, w8s4:** Overlap hurts generalisation (LP finding, +2.5–3.4 pp penalty)
- **w30s4:** Dominated by stimulus fingerprinting (84.5% of above-chance 9-class is stimulus identity); only 1 window/stimulus makes contrastive learning impossible

---

## Phase 2: LoRA Finetuning (Baseline 2)

### Goal

Establish whether LoRA adaptation improves cross-subject emotion recognition beyond frozen features. Test both pool and nopool_flat to determine whether LoRA fixes the pooling bottleneck.

### Experiments

| ID | Config | Folds | Task | Eval | Purpose |
|---|---|---|---|---|---|
| 2A | LoRA + pool, w10s10, rank=8 | 10 | binary + 9-class | Std CV + Gen (5 seeds) | Primary baseline 2 |
| 2A-cls | Same + `--unfreeze-cls` | 10 | binary + 9-class | Std CV + Gen (5 seeds) | Test cls_query adaptation |
| 2B | LoRA + pool, w10s10, rank in {4, 16} | 3 | binary | Std CV | Rank selection |
| 2C | LoRA + nopool_flat, w10s10, rank=8 | 3 | binary + 9-class | Std CV | Pool vs flat with LoRA |

### Training setup (existing code)

- Two-phase: 10 epochs head-only → LoRA unfrozen
- lr_head=1e-3, lr_lora=1e-4, batch_size=64, patience=20
- Adam optimizer, ReduceLROnPlateau

### Key comparison

| Metric | Phase 1: frozen+nopool_flat | Phase 2: LoRA+pool | Phase 2: LoRA+pool+cls | Phase 2: LoRA+nopool_flat |
|---|---|---|---|---|
| Std CV binary | 66.11% | ? | ? | ? |
| Gen binary | 57.29% | ? | ? | ? |
| Std CV 9-class | 38.02% | ? | ? | ? |
| Gen 9-class | 18.84% | ? | ? | ? |

**Phase 2 results determine Phase 3 architecture** (see below).

---

## Phase 3: LoRA + SupCon (Contribution)

### Goal

Demonstrate that joint SupCon + CE training produces embeddings that are more subject-invariant and emotion-discriminative than CE alone, as measured by the generalization protocol.

### Architecture decision (depends on Phase 2 outcome)

The aggregation strategy for Phase 3 is data-driven based on Phase 2 results:

**Scenario A: LoRA+pool closes most of the gap with LoRA+nopool_flat**

LoRA has fixed the pooling bottleneck → Use trainable cls_query_token for Phase 3

```
REVE (LoRA) → (B, 30, H, 512) → trainable attention pooling → (B, 512)
                                                                    |
                                                    +---------------+---------------+
                                                    |                               |
                                             Projection head                Classification head
                                          MLP(512->128->128)+L2norm         Linear(512, N_classes)
                                                    |                               |
                                                L_SupCon                         L_CE
```

Why: Clean comparison with Phase 2 pool (same architecture, only loss changes). SupCon teaches the pooling to preserve emotion structure.

**Scenario B: LoRA+pool still significantly underperforms LoRA+nopool_flat**

Pooling bottleneck is architectural → Phase 3 must bypass pooling

```
REVE (LoRA) → (B, 30, H, 512) → flatten → Linear(168960, 512) → ReLU → (B, 512)
                                                                              |
                                                              +---------------+---------------+
                                                              |                               |
                                                       Projection head                Classification head
                                                    MLP(512->128->128)+L2norm         Linear(512, N_classes)
                                                              |                               |
                                                          L_SupCon                         L_CE
```

Why: The learned bottleneck replaces the pre-trained cls_query_token, learning task-specific aggregation guided by both CE and SupCon. In this case, Phase 2 nopool_flat (with direct Linear(168960, N)) serves as the baseline, and Phase 3 adds both the bottleneck and SupCon.

In either scenario, the projection head is discarded at test time; classification uses the 512-D aggregated embedding.

### SupCon design choices

- **Temperature tau:** Start 0.07 (standard SupCon), sweep {0.05, 0.07, 0.1}
- **Loss weights:** L = alpha * L_CE + beta * L_SupCon. Start alpha=1.0, beta=0.5. Sweep beta in {0.25, 0.5, 1.0}
- **Batch size:** >=128 (critical for SupCon — need enough same-class pairs per batch). Use gradient accumulation if GPU memory is limiting
- **Class-balanced sampling:** Enforce equal class representation per batch
- **Training phasing:** 3-phase: head-only (5 ep) → CE+LoRA (5 ep) → full CE+SupCon+LoRA

### Experiments

| ID | Config | Folds | Task | Eval | Purpose |
|---|---|---|---|---|---|
| 3B | LoRA+SupCon, tau x beta sweep (9 combos) | 3 | binary | Std CV | HP selection |
| 3A | LoRA+SupCon, best HPs, w10s10 | 10 | binary + 9-class | Std CV + Gen (5 seeds) | Primary contribution |
| 3D | SupCon-only ablation (alpha=0, beta=1) | 3 | binary | Std CV | Loss component isolation |
| 3C | LoRA+SupCon, w5s5 | 3 | binary | Std CV | Window/data-size comparison |

---

## Compute Tiers

### Tier 1 — Tight (~1 week single GPU)

Run only the essential experiments:

| ID | Runs | Purpose |
|---|---|---|
| 2A | 10 folds x (binary + 9-class) = 20 | Primary LoRA baseline (pool) |
| 2C | 3 folds x binary = 3 | Quick pool-vs-flat check |
| 3B | 9 combos x 3 folds = 27 | HP selection for SupCon |
| 3A | 10 folds x (binary + 9-class) = 20 | Primary contribution |
| **Total** | **~70 fold-level runs** | |

Skip: rank sweep (use rank=8), cls_query ablation, generalization eval for Phase 2 (infer from Phase 1 patterns), w5s5 comparison, SupCon-only ablation.

### Tier 2 — Moderate (~2-3 weeks single GPU)

Full primary plan:

| ID | Runs | Purpose |
|---|---|---|
| 2A | 20 | LoRA+pool baseline |
| 2A-cls | 20 | cls_query adaptation test |
| 2B | 6 (2 ranks x 3 folds) | Rank sweep |
| 2C | 6 (3 folds x 2 tasks) | Pool-vs-flat |
| 2A-gen | 5 seeds x 10 folds x 2 tasks = 100 | Gen eval for best Phase 2 |
| 3B | 27 | HP sweep |
| 3A | 20 | Primary contribution |
| 3A-gen | 100 | Gen eval for Phase 3 |
| 3D | 3 | SupCon-only ablation |
| **Total** | **~302 fold-level runs** | |

### Tier 3 — Comfortable (multi-GPU or >3 weeks)

Full plan + secondary experiments:

Everything in Tier 2, plus:

| ID | Runs | Purpose |
|---|---|---|
| 3C | 6 (3 folds x 2 tasks) | w5s5 comparison |
| 2A-w5s5 | 6 | LoRA baseline at w5s5 |
| 3B-extended | 18 | Batch size + proj dim sweep |
| Gen for all | ~50 | Gen eval for secondary configs |
| **Total** | **~382 fold-level runs** | |

---

## Sequencing (dependency graph)

```
Phase 2:
  2A (LoRA+pool, 10 folds) --------------------+
  2C (LoRA+nopool_flat, 3 folds) ---------------+
  2B (rank sweep, 3 folds per rank) ------------+
  2A-cls (unfreeze cls, 10 folds) --------------+
                                                v
                              Compare pool vs flat results
                              Select best rank
                              Decide Phase 3 aggregation (Scenario A or B)
                              Run gen eval for best Phase 2 config
                                                |
                                                v
Phase 3:
  3B (tau x beta sweep, 3 folds) ---------------+
                                                v
                              Select best tau, beta
                                                |
                                                v
  3A (full 10 folds, binary + 9-class) ---------+
  3A-gen (gen eval, 5 seeds) -------------------+
  3D (SupCon-only ablation) --------------------+
  3C (w5s5 comparison, if time) ----------------+
```

---

## Evaluation Protocol (all experiments)

1. **Standard CV** (10-fold cross-subject): Accuracy, AUROC, F1 (mean +/- std across folds)
2. **Generalization** (cross-subject x cross-stimulus, 5 seeds): Accuracy, AUROC (mean +/- std across seeds)
3. **Generalization is the primary metric for claiming improvement.** Standard CV improvements alone are insufficient — they may reflect improved stimulus fingerprinting rather than emotion recognition.

---

## Critical files to modify

| File | Changes needed |
|---|---|
| `src/approaches/lora_finetuning/model.py` | Add SupCon projection head, joint loss, nopool_flat option, bottleneck layer (Scenario B) |
| `src/approaches/lora_finetuning/train_lora.py` | Add `--generalization` flag, SupCon hyperparameters (tau, alpha, beta), batch size / gradient accumulation, class-balanced sampler |
| `src/thu_ep/dataset.py` | May need class-balanced batch sampler for SupCon |
| `src/thu_ep/folds.py` | Already has generalization logic — reuse directly |
| `docs/linear_probing_results.md` | Reference: all Baseline 1 numbers that subsequent phases must beat |
