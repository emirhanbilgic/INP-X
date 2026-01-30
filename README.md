# INP-X: Inpainting Exchange

**AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange**

[![Dataset](https://img.shields.io/badge/Dataset-90K_Images-green.svg)](#dataset)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Modern deep learning-based inpainting enables realistic local image manipulation, raising critical challenges for reliable detection. However, we observe that **current detectors primarily rely on global artifacts** that appear as inpainting side effects, rather than on locally synthesized content.

This repository contains the code and dataset for our paper:

> **AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange**
> 
> Elif Nebioglu*, Emirhan Bilgiç*, Adrian Popescu
>
> *Equal Contribution

## Key Findings

- **VAE-induced artifacts**: VAE-based reconstruction induces a subtle but pervasive spectral shift across the entire image, including unedited regions
- **INP-X operation**: We introduce Inpainting Exchange (INP-X), which restores original pixels outside the edited region while preserving all synthesized content
- **Detector vulnerability**: Under INP-X, pretrained state-of-the-art detectors exhibit dramatic accuracy drops (e.g., from 91% to 55%), frequently approaching chance level
- **Improved training**: Training on INP-X images yields better generalization and localization than standard inpainting

## Method

![INP-X Pipeline](figures/teaser.png)

INP-X surgically restores original pixels outside the edited region while preserving the generated content within the mask. If detectors truly identify synthetic content, they should spot it in exchanged images since the fake content remains intact.

## Dataset

We construct a **90K-image benchmark** extending Semi-Truths across 4 datasets:
- CelebA-HQ
- CityScapes
- OpenImages
- SUN-RGBD

Each dataset includes:
1. **Real images** (x)
2. **Standard inpainted images** (x̃)
3. **Exchanged inpainted images** (x^ex)

Inpainting is performed using three models:
- Kandinsky 2.2
- OpenJourney
- Stable Diffusion v1.4

## Results

### Pretrained Detectors

| Detector | Data | Accuracy | AUC |
|----------|------|----------|-----|
| Corvi2023 | INP | 0.942 | 0.989 |
| Corvi2023 | **INP-X** | **0.554** | **0.519** |
| DNF | INP | 0.710 | 0.779 |
| DNF | **INP-X** | **0.604** | **0.643** |
| SPAI | INP | 0.661 | 0.743 |
| SPAI | **INP-X** | **0.542** | **0.567** |

### Commercial APIs

| Detector | Data | Accuracy | AUC |
|----------|------|----------|-----|
| Hive Moderation | INP | 0.914 | 0.921 |
| Hive Moderation | **INP-X** | **0.548** | **0.578** |
| Sightengine | INP | 0.926 | 0.935 |
| Sightengine | **INP-X** | **0.550** | **0.588** |

## Installation

```bash
git clone https://github.com/emirhanbilgic/INP-X.git
cd INP-X
pip install -r requirements.txt
```

## Usage

### Applying INP-X to an inpainted image

```python
import numpy as np
from PIL import Image

def inpainting_exchange(original, inpainted, mask):
    """
    Apply Inpainting Exchange operation.
    
    Args:
        original: Original image (H, W, 3)
        inpainted: Standard inpainted image (H, W, 3)
        mask: Binary mask where 1 = inpainted region (H, W)
    
    Returns:
        Exchanged image with original pixels restored outside mask
    """
    mask_3d = np.expand_dims(mask, axis=-1)
    exchanged = inpainted * mask_3d + original * (1 - mask_3d)
    return exchanged.astype(np.uint8)
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{nebioglu-bilgic2025inpx,
  title={AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange},
  author={Nebioglu, Elif and Bilgi{\c{c}}, Emirhan and Popescu, Adrian},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.