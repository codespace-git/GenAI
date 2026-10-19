# Conditional VAE for Eyeglasses Image Generation

A label-cleaning pipeline and a conditional VAE (PyTorch) that generates face images with or without eyeglasses. Built for the Generative AI course (Feb–Mar 2026).

## What's in this repo

| File | Purpose |
|---|---|
| `kagg.py` | Runs a pretrained `GlassesClassifier` (glasses-detector) over the training images. Whenever its prediction disagrees with the provided label, the image opens in an OpenCV window: press `m` to flip the label, `s` to skip, `q` to quit. Corrected labels are written to `train-corrected.csv` (autosaved every 100 rows). |
| `correction-log.csv` | Log of the manually corrected images (`image-no`, `final-label`): 529 corrections. |
| `resizing.py` | Resizes face images to 64×64 (`cv2.INTER_AREA`) into `resized_dataset/`. The image index range is hard-coded in the loop. |
| `assign_final.ipynb` | Final notebook: dataset class with Albumentations augmentations, conditional VAE, hyperparameter ablation, and evaluation (FID, LPIPS, precision/recall). |
| `assign.ipynb` | Earlier version of the notebook. |

## Data

Kaggle dataset [`jeffheaton/glasses-or-no-glasses`](https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses), downloaded with `kagglehub`. `train-corrected.csv` holds the 4,500 training rows with the corrected `glasses` labels.

## Model

- Conditional VAE on 64×64 RGB images: a 3-layer convolutional encoder and a transposed-convolution decoder.
- The class label (glasses / no glasses) goes through a 16-dimensional embedding that is concatenated to the encoder features and to the latent code, so generation can be controlled by label.
- Loss: reconstruction (MSE or L1) + β · KL divergence. Optimizer: Adam with a ReduceLROnPlateau scheduler, batch size 64.
- Augmentations (Albumentations): affine (small shift, scale, rotation), horizontal flip, brightness/contrast, Gaussian noise.

## Ablation results

Baseline: latent dim 64, lr 1e-3, β = 1e-3, LeakyReLU, MSE loss.

| Run | Change vs. baseline | FID ↓ | LPIPS |
|---|---|---|---|
| `vae_baseline` | none | 175.13 | 0.2627 |
| `vae_lr_1e-4` | lr = 1e-4 | 280.33 | 0.2904 |
| `vae_zdim_128` | latent dim = 128 | 183.96 | 0.2727 |
| `vae_act_elu` | ELU activation | 170.58 | 0.2531 |
| `vae_loss_l1` | L1 reconstruction loss | **123.54** | 0.2225 |

L1 reconstruction loss lowered FID by about 29% relative to the MSE baseline (175.13 → 123.54). The notebook also defines a KL-weight run (`vae_beta_2`); the table lists the results saved in the notebook output.

## Setup

Python ≥ 3.13. Install the dependencies in `pyproject.toml` (glasses-detector, kagglehub, matplotlib, opencv-python, pandas). `assign_final.ipynb` additionally needs `torch`, `torchvision`, `torchmetrics`, `lpips`, `albumentations` and `numpy`, which are not yet listed in `pyproject.toml`.

Typical order: download the data via `kagglehub` → `python kagg.py` (label correction) → `python resizing.py` → run `assign_final.ipynb`.
