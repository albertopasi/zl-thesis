# TODOs

VISUALIZE EACH EEG TRIAL

MANUAL (VISUAL) ANALYSIS: PROBLEMATIC SUBJ/STIMULUS

- sub 37, stim 16, 22, 25          , problem: THICK LINES
- sub 46, stim 4, 10, 18, 24, 27   , problem: channel O1
- sub 75, almost every stimuli     , problem: channels Oz and PO4

discard them first

try dropout

finetune (LoRA) with cross entropy loss only

implement generalization test (exclude stimulus, see if nopool flat 30s window setting still works well -> if not, model is learning to identify stimuli rather than emotions.)