# image_foundations
Foundational models in Image overview 

## Updates
- [23-02-2024] [Patch n Pack](https://arxiv.org/pdf/2307.06304.pdf)
- [12-02-2024] [Masked AutoEncoders](https://arxiv.org/pdf/2111.06377.pdf)
- [27-01-2024] [SimMIM: A Simple Framework for Masked Image Modelling](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.pdf)
- [20-01-2024] [AutoRegressive Image models](https://arxiv.org/pdf/2401.08541.pdf)


## Setup Guide - Local and Cloud

### Using uv (recommended)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed
- Create venv and install deps. Pick a PyTorch variant:
  - **Mac / Linux CPU**: `uv sync --extra cpu`
  - **Linux GPU (CUDA 11.8)**: `uv sync --extra cu118`
  - **Linux GPU (CUDA 12.1)**: `uv sync --extra cu121`
  - **Linux GPU (CUDA 12.4)**: `uv sync --extra cu124`
- Activate: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
- Register Jupyter kernel: `python -m ipykernel install --user --name if --display-name "Python (if)"`

### Notebooks
- Clear jupyter notebook outputs: `jupyter nbconvert --clear-output --inplace *.ipynb`