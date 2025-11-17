# Image Caption Generator

A clean, production-ready implementation of an **Image Caption Generator**: a deep-learning system that takes an input image and produces a natural-language description. This project provides training and inference code, pre-processing utilities, evaluation scripts, and an optional  for integration.

---

## Table of Contents

* [Project overview](#project-overview)
* [Features](#features)
* [Tech stack](#tech-stack)
* [Model architecture](#model-architecture)
* [Dataset](#dataset)
* [Installation](#installation)
* [Quick start — Inference](#quick-start--inference)
* [Training](#training)
* [Evaluation](#evaluation)
* [ (optional)](#rest-api-optional)
* [Project structure](#project-structure)
* [Tips & best practices](#tips--best-practices)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)

---

## Project overview

This repository implements an end-to-end Image Captioning pipeline: from dataset pre-processing and model training to inference and evaluation. The primary goal is to generate fluent, descriptive captions for images using a combination of convolutional image encoders and sequence decoders (LSTM/Transformer). It is suitable for experimentation, fine-tuning on a custom dataset, and production deployment.

## Features

* Image encoder (pretrained CNN backbone) for robust visual features
* Sequence decoder (LSTM or Transformer) that generates captions
* Tokenizer and vocabulary utilities with configurable thresholds
* Data loaders with batching, augmentation, and caching
* Training loop with checkpointing and learning-rate scheduling
* Inference script for single images and batch predictions
* Evaluation scripts (BLEU, CIDEr, METEOR — pluggable)
* Optional lightweight  for serving the model
* Config-driven setup (YAML/JSON) so experiments are reproducible

## Tech stack

* Python 3.8+
* PyTorch (recommended) or TensorFlow (optional)
* torchvision / timm for pretrained backbones
* tqdm, numpy, pandas for utilities
* Pillow for image handling
* Flask / FastAPI for the serving endpoint (optional)

## Model architecture

Two common patterns are provided:

1. **CNN + LSTM**

   * Visual features extracted with a pretrained CNN (e.g., ResNet50, EfficientNet).
   * A learned linear projection maps CNN features to the decoder embedding size.
   * LSTM decoder with embedding layer, attention mechanism (optional), and linear output to softmax over the vocabulary.

2. **Vision Transformer + Transformer decoder**

   * ViT or a CNN backbone producing patch or spatial tokens.
   * Transformer decoder stacks produce captions autoregressively.

Both implementations are modular: you can swap encoders, decoders, attention modules, or positional encodings via configuration.

## Dataset

The repository assumes COCO-style datasets (image files + captions JSON). You can use MS COCO (commonly used), Flickr8k, or Flickr30k. Utilities are provided to:

* tokenize captions and build a vocabulary with a `min_freq` threshold
* convert captions to token sequences and pad them
* create PyTorch `Dataset` and `DataLoader` instances

> **Note:** You must download the dataset yourself and set `DATASET_ROOT` in the configuration. For COCO, download images and the captions JSON files separately.

## Installation

```bash
# clone
git clone https://github.com/<your-username>/image-caption-generator.git
cd image-caption-generator

# create virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# install
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` should include packages such as `torch`, `torchvision`, `tqdm`, `Pillow`, `pyyaml`, `numpy`, `pandas`, and `transformers` (if using Transformer decoders).

## Quick start — Inference

1. Place your trained checkpoint in `checkpoints/` (e.g. `checkpoints/best.pth`).
2. Run the inference script on one image:

```bash
python scripts/infer.py \
  --image_path path/to/image.jpg \
  --checkpoint checkpoints/best.pth \
  --vocab_path data/vocab.json \
  --max_len 20
```

Example output:

```
Input: image.jpg
Output caption: "A group of people standing around a wooden table with food on it."
```

The script also supports beam search (set `--beam_size 3`) and batch inference using a directory input.

## Project structure

```
image-caption-generator/
├── app/                  # FastAPI server
├── configs/              # YAML experiment configs
├── data/                 # (place dataset and vocab here)
├── notebooks/            # EDA and qualitative checks
├── outputs/              # checkpoints, logs, visualizations
├── scripts/              # helper CLI scripts (infer, preprocess)
├── src/                  # core modules (models, datasets, utils)
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Tips & best practices

* **Start small**: train on a small subset (Flickr8k) to iterate faster.
* **Freeze backbone** initially to stabilize training, then unfreeze for fine-tuning.
* **Use beam search** at inference for better captions, but validate quality with automatic metrics and human inspection.
* **Monitor hallucinations** — sometimes decoders generate plausible but incorrect details; consider stronger visual grounding (attention, object detectors).

## Contributing

Contributions are welcome. Please open an issue for feature requests or bug reports, and submit pull requests for new models, datasets, or improvements. Follow the existing code style and add tests where appropriate.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgements

* MS COCO dataset and its contributors
* torchvision / timm and the maintainers of pretrained models

---

*Last updated: 2025-11-17*
