# DIP-SR: Deep Image Prior for Super-Resolution

This repository is a small, Colab-friendly starting point for single-image
super-resolution with a Deep Image Prior (DIP). The model is trained from
scratch for each image; there is no dataset pretraining.

The repository is intentionally notebook-first. The code under `src/dip_sr`
is a small importable package, while the committed notebook contains the
experiment and its visual and quantitative comparisons.

## Assignment-aligned workflow

1. Clone or upload this repository to Google Colab.
2. Open [`notebooks/super_resolution_template.ipynb`](notebooks/super_resolution_template.ipynb).
3. Run every cell from top to bottom in a fresh runtime.
4. Authenticate with W&B when prompted. The experiment uses the
   [`dip-sr-pnp` W&B project](https://wandb.ai/dhadheechi/dip-sr-pnp).
5. Verify the PSNR table, bar chart, image grids, and W&B runs before
   downloading the repository.

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

## Layout

```text
src/dip_sr/                  reusable data, model, metrics, and training code
notebooks/super_resolution_template.ipynb
requirements.txt             pip dependencies for local use and Colab
```

## Scope

The notebook includes a controlled variant based on the PnP ideas in
`Lecture9_RoleOfDenoisersOutsideDenoising.pdf`. It first runs the DIP baseline,
then performs image-space plug-and-play refinement: bicubic data-consistency
gradient steps are alternated with gentle TV denoising while the denoiser
strength is gradually reduced. The HR image is used only for evaluation, not
by the refinement. The comparison reports bicubic, DIP, and DIP + PnP PSNR,
along with image grids for every evaluated image.

## Colab step-by-step

1. Open `notebooks/super_resolution_template.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU** (or another available
   GPU).
3. Run all cells in order. The setup cell clones the repository if necessary
   and installs `requirements.txt`, including `wandb`.
4. Create a free account at [wandb.ai](https://wandb.ai), or use an existing
   one. Run the W&B cell and paste the API key from the displayed login link.
   The key is stored by the Colab runtime and is not written to the notebook.
5. The notebook prints the PSNR for bicubic, DIP, and DIP + PnP, plus the PnP
   improvement in dB. Open the W&B project link above to inspect the runs,
   including `dip-baseline-zebra` and `dip-pnp-tv-zebra`.

The experiment uses the images listed in the notebook to create reproducible
synthetic bicubic LR observations. Keep the image, scale, seed, and iteration
budget identical when comparing DIP with DIP + PnP.
