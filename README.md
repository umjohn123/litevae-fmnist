# LiteVAE — Fashion-MNIST Deep Autoencoder (AI Assignment #1)

> CISC3024 Pattern Recognition · AI Assignment #1 · 2026

This repository is the complete deliverable of **AI Assignment #1**: a
**LiteVAE** (NeurIPS 2024) deep autoencoder — a lightweight and efficient
variational autoencoder — implemented end-to-end by an AI assistant, evaluated
on **Fashion-MNIST** for image reconstruction and denoising. The whole workflow
(algorithm search, coding, debugging, report writing) was done by AI; no code
was written by hand.

## Repository layout

| File / folder | Description |
|---------------|-------------|
| `litevae_fmnist.py` | LiteVAE implementation: Haar-DWT encoder front-end + feature extraction/aggregation + fully-convolutional decoder, VAE losses, training & evaluation |
| `litevae_site/index.html` | An English webpage that displays the source code |
| `index.html` | Root copy of the English code-viewer page (served by GitHub Pages) |
| `output/` | Result figures: qualitative samples, training curves |
| `AI_Assignment1_Report.docx` / `.pdf` | The assignment report (Word / PDF) |
| **`requirements.txt`** | **Assignment requirements (NOT Python dependencies):** the report must cover 6 points — (1) how I asked AI tools to find the algorithm, (2) algorithm description, (3) how AI implements the algorithm, (4) experiment settings and results, (5) what I learnt from this AI assignment, and (6) a webpage link of the source codes |

> **Important:** `requirements.txt` is the assignment specification (what the
> report must contain) — it is **not** a Python dependency list. The real
> runtime dependencies are listed below.

## Dependencies (to run the code)

CPU environment is sufficient:

```bash
pip install torch torchvision numpy matplotlib pillow --index-url https://download.pytorch.org/whl/cpu
```

- Python 3.10+
- PyTorch 2.x and torchvision
- NumPy, Matplotlib, Pillow

## Run

The Fashion-MNIST dataset is downloaded automatically on first run:

```bash
python litevae_fmnist.py --epochs 14 --kl-weight 1e-4
```

Figures and the model checkpoint are saved to `output/`.

## Results (Fashion-MNIST test set)

| Metric | Clean reconstruction | Denoising (Gaussian, σ=0.2) |
|--------|----------------------|-----------------------------|
| MSE ↓  | 0.0101               | 0.0113                      |
| PSNR ↑ (dB) | 19.97            | 19.47                       |
| SSIM ↑ | 0.878                | 0.864                       |

## Source-code page (report item 6)

- GitHub repository: https://github.com/umjohn123/litevae-fmnist
- English code-viewer page: https://umjohn123.github.io/litevae-fmnist/

## Reference

S. Sadat, J. Buhmann, D. Bradley, O. Hilliges, R. M. Weber,
*LiteVAE: Lightweight and Efficient Variational Autoencoders for Latent
Diffusion Models*, NeurIPS 2024. arXiv:2405.14477.