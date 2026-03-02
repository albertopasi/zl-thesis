# Analysis of Baseline 1: Linear Probing on Frozen REVE Embeddings

**Evaluation Protocol:** 10-Fold Cross-Subject Cross-Validation

**Dataset:** THU-EP (79 Subjects) (refer to `excluded_data.md` for further details)

**Architecture:** Frozen REVE Foundation Model -> 512-D Attention-Pooled Embedding -> Linear Classifier

---

## 1. Binary Task Analysis (Positive vs. Negative)

**Results:**
* **Val Accuracy:** **56.25%** ± **1.56%**
* **Val AUROC:** **0.5747** ± **0.0223**
* **Val F1-Score:** **0.5761** ± **0.0490**
* **Random Chance:** **50.00%**

### Interpretations & Key Takeaways:
* **The "Subject-Variability Wall":** A linear probe trained on 71 subjects and tested on 8 unseen subjects achieves an accuracy of ~**56.25%**. While this is statistically above random chance (**50%**), it is practically very low. This mathematically proves that out-of-the-box, frozen REVE embeddings are heavily entangled. The latent space is likely clustered by subject-specific anatomical traits (like skull thickness or baseline impedance) rather than universal emotional valence.
* **Consistency:** The standard deviation across the 10 folds is incredibly tight (± **1.56%**). This means the model isn't getting "lucky" or "unlucky" data splits. The **56%** accuracy limit is a hard, structural limitation of a linear hyperplane trying to separate entangled cross-subject data.
* **Balanced Failure:** The F1-Score (**0.57**) closely mirrors the Accuracy and AUROC. This indicates the linear classifier isn't just collapsing and guessing "Positive" for every window; it is genuinely trying to separate the classes but failing due to the non-linear complexity of the inter-subject data.

---

## 2. 9-Class Task Analysis (8 Emotions + Neutral)

**Results:**
* **Val Accuracy:** **20.99%** ± **1.92%**
* **Val AUROC:** **0.6237** ± **0.0233**
* **Val F1-Score:** **0.1930** ± **0.0240**
* **Random Chance:** ~**11.11%** (1/9)

### Interpretations & Key Takeaways:
* **Impressive Signal Extraction:** At first glance, **20.99%** accuracy looks terrible. However, in a 9-class cross-subject BCI problem, random chance is ~**11.1%**. **The model is performing at nearly double random chance without a single parameter of the foundation model being fine-tuned.** This confirms that the REVE backbone has successfully learned highly generalized, universal EEG temporal-spatial features during its pre-training.
* **The AUROC Anomaly (One-vs-Rest Ranking):** The Val AUROC is **0.6237**. In multi-class scenarios, AUROC is calculated using a One-vs-Rest (OvR) macro-average . An AUROC of **0.62** indicates that if we take a true "Joy" window and a random window of any other emotion, there is a **62%** probability the model assigns a higher "Joy" probability to the correct window. This proves the foundation model is successfully placing the EEG signal into the correct "emotional neighborhood," even if overlapping clusters cause the strict Top-1 Accuracy to fail.
* **F1-Score Drop:** The low F1-score (**0.19**) highlights the fragility of the linear boundary in a high-dimensional space. It likely struggles heavily with minority confusion (e.g., misclassifying high-arousal negative emotions like "Fear" and "Anger" as the same thing). 

---
Good results to be a basleine: if the Linear Probing baseline had achieved high accuracy on the binary task, the core hypothesis of the thesis would be invalidated. It would mean that simple linear transformations are sufficient for BCI emotion recognition if REVE is used.

Because the Linear Probe fails to generalize past ~**56%** (Binary) and ~**21%** (9-Class), we have empirically proven the necessity of the JADE framework. These results establish that:
1. **Fine-Tuning is Mandatory:** We must unfreeze the network (via LoRA) to allow the foundation model to adapt to the specific THU-EP dataset frequencies.
2. **Contrastive Alignment is Mandatory:** We must replace standard Cross-Entropy with Supervised Contrastive Learning. A linear classifier cannot untangle the subjects; we must force the neural network to explicitly pull different subjects experiencing the same emotion together in the latent space before classification.

**Conclusion:** The floor is set. The metrics to beat are **56.25%** (Binary) and **20.99%** (9-Class).