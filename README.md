# Closed-Form Flow Matching

MVA Generative Modelling project based on:

> **On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity**
> Bertrand, Gagneux, Massias, Emonet (2025) — [arXiv:2506.03719](https://arxiv.org/abs/2506.03719)

## Project overview

This project investigates the sources of generalization in flow matching models through:

1. **Stochasticity analysis** (Figure 1) — Measure the alignment between the optimal velocity field û★ and the conditional target u^cond on toy and image datasets across time and dimension.
2. **Velocity approximation failure** (Figure 2) — Train flow matching models on varying dataset sizes and compare the learned velocity u_θ to the closed-form û★.
3. **Generalization switching time** (Figure 3) — Build hybrid models (û★ then u_θ) to pinpoint when generalization arises during the flow.
4. **Image experiments** — Train Vanilla CFM and Empirical Flow Matching (EFM) on MNIST/Fashion-MNIST and compare.

## Structure

```
src/
├── data/
│   ├── toy.py              # Toy datasets (moons, rings, Gaussian mixtures)
│   └── images.py           # MNIST / Fashion-MNIST loaders
├── models/
│   ├── mlp.py              # MLP velocity network for 2D experiments
│   └── unet.py             # Small UNet for image experiments
├── flow_matching/
│   ├── closed_form.py      # Closed-form velocity û★ (Proposition 1)
│   ├── cfm.py              # Vanilla Conditional Flow Matching (Algorithm 1)
│   ├── efm.py              # Empirical Flow Matching (Algorithm 2)
│   └── sampler.py          # ODE integration for generation
├── metrics/
│   └── evaluation.py       # Cosine similarity, FID, nearest-neighbor distance
notebooks/
├── 01_figure1_stochasticity.ipynb
├── 02_figure2_velocity_approx.ipynb
├── 03_figure3_generalization_time.ipynb
└── 04_image_experiments.ipynb
```

## Usage

```bash
pip install -r requirements.txt
```

All experiments are designed to run on a free Google Colab GPU (T4).
