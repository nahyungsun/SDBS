# Semantic Diverse Beam Search (SDBS)

[![Paper](https://img.shields.io/badge/Paper-Knowledge--Based%20Systems-blue)](https://www.sciencedirect.com/science/article/abs/pii/S095070512501439X)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-green.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12-red.svg)](https://pytorch.org/)

Official implementation of **"Augmented Decoding Method Using Semantic Diverse Beam Search for Language Generation Model"** (Knowledge-Based Systems, 2025)

## Overview

Image captioning models often generate repetitive and semantically similar captions despite appearing lexically different. Existing diversity-oriented decoding methods rely on surface-level string matching, incorrectly treating phrases like "dog runs" and "canine sprints" as diverse outputs.

**SDBS (Semantic Diverse Beam Search)** addresses this limitation by operating in semantic space rather than lexical space, using knowledge graph-based similarity to achieve genuine semantic diversity.

![SDBS Comparison](https://github.com/nahyungsun/SDBS/assets/54011107/c9d383ae-0ab8-4ca4-a3ee-4953bd9c4414)

## Key Contributions

- **Semantic Diversity Scoring**: Knowledge graph-based (WordNet) similarity replaces string-based Hamming similarity
- **Adaptive Thresholding**: 0.8 similarity threshold eliminates unnecessary computations for semantically distant word pairs
- **Stratified Top-k Sampling**: Time-step aware sampling based on part-of-speech distributions
- **Beam Size Normalization**: Decouples beam size from group size for flexible hyperparameter selection
- **Early-Stop Strategy**: Reduces computational complexity while maintaining generation quality

## Results

### Performance Comparison (COCO 2014)

| Model | Decoding | ALLSPICE | SPICE | CIDEr | Self-CIDEr | BLEU-4 | METEOR |
|-------|----------|----------|-------|-------|------------|--------|--------|
| Transformer | Beam Search | 0.199 | 0.191 | 1.124 | 0.768 | 0.244 | 0.291 |
| Transformer | DBS | 0.270 | 0.246 | 1.353 | 0.823 | 0.343 | 0.322 |
| Transformer | **SDBS** | **0.275** | **0.255** | **1.429** | 0.832 | **0.367** | **0.337** |
| A2I2 | Beam Search | 0.170 | 0.170 | 1.061 | 0.000 | 0.195 | 0.260 |
| A2I2 | DBS | 0.255 | 0.233 | 1.407 | 0.798 | 0.345 | 0.316 |
| A2I2 | **SDBS** | **0.259** | **0.236** | 1.369 | 0.778 | 0.339 | **0.322** |

### Qualitative Example

**Ground Truth Labels:**
- a man with a red helmet on a small moped on a dirt road
- a man riding on the back of a motorcycle
- a dirt path with a young person on a motor bike...

**Beam Search** (repetitive):
- a man riding a motorcycle down a dirt road
- a man riding a motorcycle on a dirt road
- a man rides a motorcycle down a dirt road

**SDBS** (semantically diverse):
- a man riding a motorcycle down a dirt road
- there is a man riding a motorcycle down a dirt road
- man riding motorcycle on dirt road with mountain in background
- the person is riding a motorcycle down the dirt road
- an image of a man riding his motor bike

## Installation

### Requirements

- Ubuntu 20.04.4 LTS
- Python 3.8.13
- PyTorch 1.12.0
- CUDA 11.6
- cuDNN 8.4.0

### Setup

```bash
# Clone repository
git clone https://github.com/nahyungsun/SDBS.git
cd SDBS

# Install dependencies
pip install nltk==3.7
pip install h5py==3.7.0
pip install lmdbdict==0.2.2
pip install scikit-image==0.19.2
pip install matplotlib==3.5.1
pip install gensim==4.2.0
pip install pyemd==0.5.1
pip install pandas
```

## Usage

### Evaluation with SDBS

Run the evaluation script with your preferred model:

```bash
# For Transformer model
bash my_gen_eval_n_dbs_.sh trans

# For A2I2 model
bash my_gen_eval_n_dbs_.sh a2i2
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--beam_size` | Beam size | 5 |
| `--sample_n` | Group size | 5 |
| `--diversity_lambda` | Diversity strength (λ) | 3.0 |
| `--sim` | Similarity threshold | 0.8 |
| `--topk1` | Stratified top-k (first steps) | 9 |
| `--topk2` | Stratified top-k (remaining steps) | 2 |
| `--apply` | Early-stop time step (τ) | 13 |
| `--dbs_type` | 1: SDBS, 2: DBS (Hamming) | 1 |

### Example Command

```bash
python eval.py \
    --batch_size 1 \
    --diversity_lambda 3.0 \
    --image_root /data/coco/ \
    --model log_trans/model-best.pth \
    --beam_size 5 \
    --sample_n 5 \
    --sim 0.8 \
    --apply 13 \
    --topk1 9 \
    --topk2 2 \
    --dbs_type 1 \
    --device cuda
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{na2025augmented,
  title={Augmented Decoding Method Using Semantic Diverse Beam Search for Language Generation Model},
  author={Na, HyungSun and Jun, Hee-Gook and Ahn, Jinhyun and Im, Dong-Hyuk},
  journal={Knowledge-Based Systems},
  pages={114400},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.knosys.2025.114400}
}
```

## Acknowledgements

This implementation is based on [ruotianluo/ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch). We thank the authors for their excellent codebase.

This research was supported by the Research Grant of Kwangwoon University in 2022.

## License

This project is for academic research purposes. Please refer to the original repository for license details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.