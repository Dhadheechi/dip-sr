# DIP-SR: Deep Image Prior for Super-Resolution

This repository is a small, Colab-friendly starting point for single-image
super-resolution with a Deep Image Prior (DIP). The model is trained from
scratch for each image; there is no dataset pretraining.

The repository is intentionally notebook-first. The code under `src/dip_sr`
is a small importable package, while the committed notebook is the cloud
experiment template that can later contain the final DIP variants and W&B
results.

## Assignment-aligned workflow

1. Clone or upload this repository to Google Colab or Kaggle.
2. Open [`notebooks/super_resolution_template.ipynb`](notebooks/super_resolution_template.ipynb).
3. Run the first setup cell to install the packages from `requirements.txt`.
3. Run the baseline against the included/selected image.
4. Add model or optimization variants as separate W&B runs with the same
   input, scale, seed, and evaluation protocol.
5. Commit the notebook after the cloud runs so the visual comparison, PSNR
   table, and W&B links are part of the GitHub history.

The experiment logs both low-resolution reconstruction PSNR (the optimization
objective) and high-resolution PSNR (available when a ground-truth HR image is
known). For a real inverse problem, use the former for monitoring and reserve
the latter for evaluation.

## Install

The project uses a `src` layout and a standard pip requirements file:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

In Colab, the notebook setup cell does the equivalent:

```python
!pip install -r requirements.txt
```

The requirements file includes PyTorch. Colab users can omit reinstalling
`torch` and `torchvision` if the runtime already provides compatible versions.
For GPU training, use a runtime whose PyTorch build matches its CUDA runtime.
W&B is optional at runtime: leave `wandb_project=None` for a local/offline
smoke test, or set a project name and authenticate with `wandb.login()` before
training.

## Layout

```text
src/dip_sr/                  reusable data, model, metrics, and training code
notebooks/super_resolution_template.ipynb
requirements.txt             pip dependencies for local use and Colab
```

## Scope

The notebook now includes a controlled improvement based on the PnP ideas in
`Lecture9_RoleOfDenoisersOutsideDenoising.pdf`. It first runs the DIP baseline,
then performs image-space plug-and-play refinement: bicubic data-consistency
gradient steps are alternated with TV denoising while the denoiser strength is
gradually reduced. The HR image is used only for evaluation, not by the
refinement.

## Colab step-by-step

1. Open `notebooks/super_resolution_template.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU** (or another available
   GPU).
3. Run the setup cell. It clones the repository if necessary and installs
   `requirements.txt`, including `wandb`.
4. Create a free account at [wandb.ai](https://wandb.ai), or use an existing
   one. Run the W&B cell and paste the API key from the displayed login link.
   The key is stored by the Colab runtime and is not written to the notebook.
5. Run the data, baseline, PnP, and comparison cells in order. The notebook
   prints the PSNR for bicubic, DIP, and DIP + PnP, plus the PnP improvement in
   dB.
6. Open the `dip-sr-pnp` project in W&B and compare the
   `dip-baseline-zebra` and `dip-pnp-tv-zebra` runs.

The default experiment uses `data/sr/zebra_GT.png` to create a reproducible
synthetic bicubic LR observation. For another image, change `image_path` in
the notebook and keep the image, scale, seed, and iteration budget identical
between the two runs. The UV and GitHub integration steps are intentionally
left out; GitHub is only needed to obtain the notebook.
