# qwen2.5-0.5B-alpaca-finetune

Fine-tuned Qwen2.5-0.5B on the Alpaca dataset using LoRA via HuggingFace Transformers. Conducted two experiments to investigate the effect of learning rate scheduling on training stability.

## Experiments

**Experiment 1** — baseline, lr=2e-4, 1 epoch

- Training loss: 2.09 → 1.43
- Loss std: 0.171

**Experiment 2** — cosine LR scheduler + linear warmup (warmup_ratio=0.1)

- Training loss: 2.09 → 1.43
- Loss std: 0.188
- Minimal difference observed, likely because 63 steps is too short for scheduler effects to manifest

## Observations

- Per-step loss oscillation is characteristic of small-batch training (batch size 4) on a small model
- Moving average confirms stable downward convergence in both runs
- Trained for 1 epoch only; preliminary experiments with 2 epochs showed catastrophic forgetting, where the model over-fit to Alpaca's instruction format and lost general knowledge

## Setup

- Model: Qwen2.5-0.5B
- Dataset: Alpaca
- LoRA: r=16, alpha=32, dropout=0.1, target=all-linear
- Hardware: Google Colab free tier (T4)
