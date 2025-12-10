📘 DTOF Deep Learning Pipeline for Optical Property Inversion

This repository implements a complete, reproducible deep-learning framework for estimating absorption (μa) and reduced scattering (μs′) from Monte Carlo–simulated DTOFs.

🔍 Project Overview

Time-Domain Near-Infrared Spectroscopy (TD-NIRS) captures Distribution of Time-of-Flight (DTOF) curves that encode tissue optical properties.
This project builds a CNN-based inversion model trained on MCX-simulated DTOFs to recover underlying optical properties.

The pipeline includes:

Full data preprocessing and normalisation

Multi-channel DTOF construction (raw, temporal masks, hybrid)

A flexible CNN architecture with auto-detected flattening dimension

A complete training loop with validation, checkpointing, and GPU support

An evaluation module providing MAE / RMSE metrics

A structured instruction manual describing reproducible usage

🧱 Core System Components
1. DTOFDataset

Handles the full preprocessing workflow:

Load DTOFs from CSV

Extract (μa, μs′) labels from column headers

Apply Savitzky–Golay filtering

Clip negative floating-point noise

Standardise each DTOF to zero mean and unit variance

Construct 1, 3, or 4 input channels via:

Raw DTOF

Early/Mid/Late temporal masks

Combined hybrid features

Output per sample:

signal → (C, T)   # channels × time samples  
target → (μa, μs′)

2. CNN Architecture

A domain-inspired 1D convolutional network consisting of:

Three Conv1d → BatchNorm → ReLU → MaxPool blocks

Automatic flatten-size detection via dummy forward pass

Fully connected regressor head producing:

[μa, μs′]

The architecture supports variable input channels (1, 3, or 4).

3. Training Infrastructure

Features:

PyTorch training loop

Train/validation dataloaders

MSE loss over (μa, μs′)

Adam optimiser

GPU/CPU device selection

Best-model checkpointing (best_dtof_cnn.pth)

Loss curve logging and plotting

Output of epoch-wise training + validation losses

4. Evaluation Module

The ModelEvaluator collects:

Prediction vectors across validation set

Ground-truth labels

MAE for μa and μs′

RMSE for μa and μs′

Optional sample-prediction previews

MAPE is computed internally but not used due to instability near small μa values.


