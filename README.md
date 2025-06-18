## Project Plan

We have already created a pipeline:

- **Evaluate** the performance of `wav2vec2` across different datasets (e.g., **Kinyarwanda** or **Wolof**).
- **Adopt** two training approaches:
  - [`fairseq`](https://github.com/facebookresearch/fairseq)
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **Visualize** some learned feature representations for analysis and comparison.

### Dataset

- [`mbazaNLP/kinyarwanda-tts-dataset`](https://huggingface.co/datasets/mbazaNLP/kinyarwanda-tts-dataset) *(used by Yue)*

### Model

- [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base)  
  - Quantized version: **fp16**
