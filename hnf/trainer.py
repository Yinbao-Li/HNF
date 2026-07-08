# -*- coding: utf-8 -*-
"""Part 7: HNFTrainer and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from hnf.layers import HuygensWaveLayer


@dataclass
class HNFConfig:
    input_dim: int = 2
    hidden_dim: int = 128
    output_dim: int = 1
    num_layers: int = 2
    gamma: float = 0.5
    omega: float = 0.3
    causal: bool = True
    wave_speed: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 32
    use_gpu: bool = True
    log_interval: int = 10
    train_split: float = 0.8
    kernel_regularization: float = 1e-6


class HNFTrainer:
    """惠更斯神经场训练器."""

    def __init__(
        self,
        config: HNFConfig,
        model: Optional[nn.Module] = None,
        verbose: bool = True,
    ):
        self.config = config
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        self.model = model if model is not None else self._build_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01,
        )
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def _build_model(self) -> nn.Module:
        cfg = self.config

        class HNFModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                dims = [cfg.input_dim] + [cfg.hidden_dim] * cfg.num_layers + [cfg.output_dim]
                for i in range(len(dims) - 1):
                    self.layers.append(
                        HuygensWaveLayer(
                            dims[i],
                            dims[i + 1],
                            gamma=cfg.gamma * (0.8 ** i),
                            omega=cfg.omega * (1.1 ** i),
                            causal=cfg.causal,
                            wave_speed=cfg.wave_speed * (0.9 ** i),
                            learnable_kernel_params=(i < cfg.num_layers // 2),
                        )
                    )
                    if i < len(dims) - 2:
                        self.layers.append(nn.GELU())
                        self.layers.append(nn.Dropout(0.1))

            def forward(self, x, t=None, rho=None):
                h = x
                for layer in self.layers:
                    h = layer(h, t, rho) if isinstance(layer, HuygensWaveLayer) else layer(h)
                return h

        return HNFModel()

    def train_step(self, x, y, t=None, rho=None) -> float:
        x, y = x.to(self.device), y.to(self.device)
        if t is not None:
            t = t.to(self.device)
        if rho is not None:
            rho = rho.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(x, t, rho)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def eval_step(self, x, y, t=None, rho=None) -> Tuple[float, float]:
        x, y = x.to(self.device), y.to(self.device)
        if t is not None:
            t = t.to(self.device)
        if rho is not None:
            rho = rho.to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x, t, rho)
            loss = F.mse_loss(y_pred, y)
        return loss.item(), 0.0

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        t_train=None,
        t_val=None,
        rho_train=None,
        rho_val=None,
        save_best: bool = True,
        save_path: str = "hnf_best.pt",
    ) -> Dict:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(self.config.num_epochs):
            train_loss = sum(
                self.train_step(batch_x, batch_y, t_train, rho_train)
                for batch_x, batch_y in train_loader
            ) / len(train_loader)

            val_loss = 0.0
            if X_val is not None and y_val is not None:
                val_loss, _ = self.eval_step(X_val, y_val, t_val, rho_val)

            self.scheduler.step()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if self.verbose and (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
                )

        if save_best and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            torch.save(best_model_state, save_path)
            if self.verbose:
                print(f"Best model saved to {save_path}")
        return self.history

    def predict(self, x, t=None, rho=None):
        x = x.to(self.device)
        if t is not None:
            t = t.to(self.device)
        if rho is not None:
            rho = rho.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(x, t, rho)

    def plot_history(self, figsize=(12, 4)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        axes[0].plot(self.history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)
        if self.history["train_acc"]:
            axes[1].plot(self.history["train_acc"], label="Train Acc")
            axes[1].plot(self.history["val_acc"], label="Val Acc")
            axes[1].legend()
            axes[1].grid(True)
        plt.tight_layout()
        plt.show()
