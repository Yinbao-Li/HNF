# -*- coding: utf-8 -*-
"""Huygens Neural Field (HNF) v2.0 — user's complete framework."""

from hnf.kernel import HuygensKernel
from hnf.layers import HuygensWaveLayer, HuygensAttention
from hnf.fmm import FastMultipoleMethod, DirectPropagation
from hnf.density import DensityNet
from hnf.deep import DeepHuygensKernel
from hnf.bayesian import BayesianHNF, BayesianHNFConfig
from hnf.trainer import HNFConfig, HNFTrainer
from hnf.field import HuygensNeuralField, solve_weights
from hnf.picking_model import STEADHNFPickingModel
from hnf.demos import (
    demo_causality,
    demo_classification,
    demo_long_sequence,
    demo_bayesian,
    demo_fmm_benchmark,
)
from hnf.utils import generate_wave_data, plot_kernel_matrix, compute_metrics
from hnf.data_generator import (
    FieldDataset,
    generate_plane_wave,
    generate_radial_wave,
    generate_vortex_field,
    sample_sparse_observations,
    build_synthetic_sample,
)
from hnf.visualize import plot_reconstruction, plot_field_comparison, plot_observation_distribution

__all__ = [
    "HuygensKernel",
    "HuygensWaveLayer",
    "HuygensAttention",
    "FastMultipoleMethod",
    "DirectPropagation",
    "DensityNet",
    "DeepHuygensKernel",
    "BayesianHNF",
    "BayesianHNFConfig",
    "HNFConfig",
    "HNFTrainer",
    "HuygensNeuralField",
    "solve_weights",
    "STEADHNFPickingModel",
    "demo_causality",
    "demo_classification",
    "demo_long_sequence",
    "demo_bayesian",
    "demo_fmm_benchmark",
    "generate_wave_data",
    "plot_kernel_matrix",
    "compute_metrics",
    "FieldDataset",
    "generate_plane_wave",
    "generate_radial_wave",
    "generate_vortex_field",
    "sample_sparse_observations",
    "build_synthetic_sample",
    "plot_reconstruction",
    "plot_field_comparison",
    "plot_observation_distribution",
]

__version__ = "2.0.0"
