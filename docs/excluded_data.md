# Manual Artifact Rejection and Data Exclusion

Before and after the preprocessing pipeline, a visual inspection of every individual EEG trial was conducted to identify severe, unrecoverable technical anomalies. Data exhibiting extreme signal degradation that could not be safely corrected via standard spatial or temporal filtering was strictly excluded to prevent the introduction of corrupt gradients during foundation model training.

Based on this manual visual analysis, the following data points were permanently removed from the dataset:

* **Subject 75 (Entire Recording):** The entire subject was discarded due to massive, continuous signal corruption across critical posterior channels (**Oz** and **PO4**). This reduced the total viable subject pool from 80 to 79.
* **Subject 37 (Specific Stimuli):** Stimuli **16, 22, and 25** were excluded. Visual inspection revealed severe high-variance artifacts (appearing as abnormally thick, dense signal bands), indicative of excessive non-neural noise or muscular interference during these specific trials.
* **Subject 46 (Specific Stimuli):** Stimuli **4, 10, 18, 24, and 27** were excluded due to severe, localized signal corruption strictly isolated to the **O1** occipital electrode.

**Impact on Pipeline:** The removal of specific stimuli for Subjects 37 and 46 results in dynamic trial counts for these individuals. The PyTorch `Dataset` and `BatchSampler` are explicitly engineered to handle these flattened, uneven index distributions without compromising the mathematical class balance required for cross-validation.
