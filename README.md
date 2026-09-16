# LiteVAE — Lightweight Variational Autoencoder for Fashion-MNIST

**AI Assignment #1 · CISC3024 Pattern Recognition · September 2026**

Deep Autoencoder implementation, generated end-to-end by an AI assistant.

## Overview

This project implements **LiteVAE**, a lightweight and efficient variational
autoencoder introduced in:

> S. Sadat, J. Buhmann, D. Bradley, O. Hilliges, R. M. Weber,
> *"LiteVAE: Lightweight and Efficient Variational Autoencoders for Latent
> Diffusion Models,"* NeurIPS 2024. ([arXiv:2405.14477](https://arxiv.org/abs/2405.14477))

The core idea: the encoder is front-loaded with a **2D Haar Discrete Wavelet
Transform (DWT)** so that the network can extract a rich, compact, image-like
representation with far fewer parameters than a standard VAE encoder. The
remaining bottleneck is a lightweight feature-extraction / aggregation module,
and the image is reconstructed by a purely convolutional decoder.

We apply LiteVAE to **image reconstruction and denoising** on the
**Fashion-MNIST** dataset as a computer-vision / pattern-recognition task.

## Files

| File | Description |
|------|-------------|
| `litevae_fmnist.py` | Full implementation: Haar DWT, DWT encoder, convolutional decoder, VAE losses, training & evaluation |
| `litevae_site/index.html` | English webpage presenting this source code |
| `output/litevae_fmnist.png` | Visual sample: original / noisy / reconstructed |
| `output/litevae_fmnist_curves.png` | Training curves (reconstruction & KL loss) |
| `requirements.txt` | Python dependencies |

## Dependencies

- Python 3.10+
- PyTorch (CPU is sufficient) & torchvision
- NumPy, Matplotlib, Pillow

Install:

```bash
pip install torch torchvision numpy matplotlib pillow --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

Run training + evaluation on Fashion-MNIST (downloads the dataset on first run):

```bash
python litevae_fmnist.py --epochs 14 --kl-weight 1e-4 --noise-std 0.2
```

Figures and the model checkpoint are written to `output/`.

## Results (Fashion-MNIST test set)

| Metric | Clean reconstruction | Denoising (Gaussian, σ=0.2) |
|--------|----------------------|-----------------------------|
| MSE    | 0.0101               | 0.0113                      |
| PSNR   | 19.97 dB             | 19.47 dB                    |
| SSIM   | 0.878                 | 0.864                       |

## Reference

S. Sadat, J. Buhmann, D. Bradley, O. Hilliges, R. M. Weber. *LiteVAE:
Lightweight and Efficient Variational Autoencoders for Latent Diffusion Models.*
38th Conference on Neural Information Processing Systems (NeurIPS 2024).