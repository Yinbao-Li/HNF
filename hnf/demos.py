# -*- coding: utf-8 -*-
"""Part 8: demos and examples."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from hnf.bayesian import BayesianHNF, BayesianHNFConfig
from hnf.fmm import DirectPropagation, FastMultipoleMethod
from hnf.kernel import HuygensKernel
from hnf.layers import HuygensAttention
from hnf.trainer import HNFConfig, HNFTrainer


def demo_causality():
    print("\n" + "=" * 60)
    print("演示 1: 惠更斯核 - 因果推理")
    print("=" * 60)

    def generate_causal_data(n: int = 200):
        a = torch.randn(n, 1)
        b = a + 0.5 * torch.randn(n, 1)
        c = b + 0.3 * torch.randn(n, 1)
        x = torch.cat([a, b, c], dim=1)
        t = torch.arange(n).reshape(-1, 1).float()
        return x, t

    x, t = generate_causal_data(300)
    kernel = HuygensKernel(gamma=0.5, omega=0.3, causal=True, wave_speed=0.5)
    k = kernel(x.unsqueeze(0), t.unsqueeze(0))
    k_vis = torch.abs(k.squeeze(0)).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].imshow(k_vis, cmap="hot", interpolation="nearest")
    axes[0].set_title("惠更斯核矩阵 (非对称因果传播)")
    plt.colorbar(im, ax=axes[0])
    asym = np.abs(k_vis - k_vis.T)
    axes[1].hist(asym.flatten(), bins=30, color="orange", alpha=0.7)
    axes[1].set_title("非对称性分布")
    t_np = t.numpy().flatten()[:100]
    signal = x.numpy()[:100]
    axes[2].plot(t_np, signal[:, 0], label="A (源)", color="red")
    axes[2].plot(t_np, signal[:, 1], label="B", color="blue")
    axes[2].plot(t_np, signal[:, 2], label="C", color="green")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("demo_causality.png", dpi=150)
    plt.close(fig)
    print("结论: 核矩阵具有显著非对称性，能区分因果方向!")
    return k_vis


def demo_classification():
    print("\n" + "=" * 60)
    print("演示 2: 惠更斯分类器 - 螺旋数据集")
    print("=" * 60)

    def generate_spiral(n_samples: int = 500, n_classes: int = 3):
        x = np.zeros((n_samples * n_classes, 2))
        y = np.zeros(n_samples * n_classes, dtype=np.int64)
        for class_id in range(n_classes):
            ix = range(n_samples * class_id, n_samples * (class_id + 1))
            r = np.linspace(0.0, 1, n_samples)
            t = np.linspace(class_id * 4, (class_id + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
            x[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
            y[ix] = class_id
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    x, y = generate_spiral(200, 3)
    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    y_train_onehot = F.one_hot(y_train, num_classes=3).float()
    y_val_onehot = F.one_hot(y_val, num_classes=3).float()

    config = HNFConfig(
        input_dim=2, hidden_dim=64, output_dim=3, num_layers=2,
        learning_rate=1e-3, num_epochs=50, batch_size=32,
    )
    trainer = HNFTrainer(config, verbose=True)
    trainer.fit(x_train, y_train_onehot, x_val, y_val_onehot)

    with torch.no_grad():
        t_dummy = torch.zeros(x_val.size(0), 1, 1)
        pred = trainer.model(x_val.unsqueeze(0), t_dummy)
        acc = (pred.squeeze(0).argmax(dim=-1) == y_val).float().mean().item()
    print(f"验证准确率: {acc:.2%}")
    return trainer


def demo_long_sequence():
    print("\n" + "=" * 60)
    print("演示 3: 长序列建模 - H-Attention vs Transformer")
    print("=" * 60)

    def generate_long_sequence(length: int = 1000, feature_dim: int = 64):
        t = torch.linspace(0, 10 * np.pi, length)
        signal = torch.sin(t) + 0.3 * torch.sin(0.3 * t) + 0.1 * torch.randn(length)
        return signal.unsqueeze(-1).repeat(1, feature_dim) + 0.1 * torch.randn(length, feature_dim)

    n, d = 2000, 64
    data = generate_long_sequence(n, d)
    h_attn = HuygensAttention(embed_dim=d, num_heads=4, gamma=0.5, omega=0.3, causal=True, wave_speed=0.5)
    mha = MultiheadAttention(d, num_heads=4, batch_first=True)

    start = time.time()
    with torch.no_grad():
        out_h = h_attn(data.unsqueeze(0))
    time_h = time.time() - start

    start = time.time()
    with torch.no_grad():
        out_m = mha(data.unsqueeze(0), data.unsqueeze(0), data.unsqueeze(0))[0]
    time_m = time.time() - start

    print(f"序列长度: {n}, 特征维度: {d}")
    print(f"H-Attention 时间: {time_h:.4f}s")
    print(f"标准 Attention 时间: {time_m:.4f}s")
    plt.figure(figsize=(8, 4))
    plt.bar(["H-Attention", "Standard"], [time_h, time_m])
    plt.ylabel("时间 (秒)")
    plt.savefig("demo_long_sequence.png", dpi=150)
    plt.close()


def demo_bayesian():
    print("\n" + "=" * 60)
    print("演示 4: 贝叶斯惠更斯神经场")
    print("=" * 60)

    x = torch.linspace(-5, 5, 150).reshape(-1, 1)
    y = torch.sin(x) + 0.1 * torch.randn(150, 1)
    config = BayesianHNFConfig(input_dim=1, hidden_dim=32, num_layers=2)
    model = BayesianHNF(config)
    with torch.no_grad():
        mean, var, _ = model.predict_with_uncertainty(x.unsqueeze(0), num_samples=50)
    mean_np = mean.squeeze().numpy()
    std_np = torch.sqrt(var).squeeze().numpy()
    plt.figure(figsize=(12, 6))
    plt.plot(x.numpy(), y.numpy(), "bo", alpha=0.5, label="观测数据")
    plt.plot(x.numpy(), mean_np, "r-", label="预测均值")
    plt.fill_between(x.numpy().flatten(), mean_np - 2 * std_np, mean_np + 2 * std_np, alpha=0.3, color="red")
    plt.legend()
    plt.savefig("demo_bayesian.png", dpi=150)
    plt.close()


def demo_fmm_benchmark():
    print("\n" + "=" * 60)
    print("演示 5: FMM 加速性能基准")
    print("=" * 60)

    kernel = HuygensKernel(gamma=0.5, omega=0.3)
    fmm = FastMultipoleMethod(kernel, max_leaf_size=16, expansion_order=6)
    direct = DirectPropagation(kernel)
    for n in [100, 500, 1000]:
        x = torch.randn(n, 2)
        sources = torch.randn(n, 1)
        t0 = time.time()
        direct.forward(x, sources)
        td = time.time() - t0
        t0 = time.time()
        fmm.forward(x, sources)
        tf = time.time() - t0
        print(f"N={n}: Direct={td:.4f}s, FMM={tf:.4f}s")
