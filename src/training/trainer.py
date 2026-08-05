from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import compute_classification_metrics


class Trainer:
    """Training loop for multimodal DTI models.

    Handles training, evaluation, early stopping, LR scheduling,
    and model checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_filename: str = "best_model.pt",
        patience: int = 8,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        grad_accum_steps: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_filename = checkpoint_filename
        self.patience = patience
        self.grad_accum_steps = grad_accum_steps

        # Speed up convolution operations with consistent input sizes
        torch.backends.cudnn.benchmark = True

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=scheduler_patience, factor=scheduler_factor
        )

    # ------------------------------------------------------------------
    # Single-epoch routines
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Train", leave=False, disable=not sys.stdout.isatty())

        self.optimizer.zero_grad(set_to_none=True)
        for step, (smiles_batch, protein_batch, y) in enumerate(pbar):
            smiles_batch = _to_device(smiles_batch, self.device)
            protein_batch = _to_device(protein_batch, self.device)
            y = y.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits = self.model(smiles_batch, protein_batch).squeeze(1)
                loss = self.criterion(logits, y)
                # Scale loss for gradient accumulation
                if self.grad_accum_steps > 1:
                    loss = loss / self.grad_accum_steps

            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(loader):
                nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad), 1.0
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * (
                self.grad_accum_steps if self.grad_accum_steps > 1 else 1
            )

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run evaluation. Returns dict with loss + classification metrics."""
        self.model.eval()
        total_loss = 0.0
        all_labels: list[float] = []
        all_probs: list[float] = []

        pbar = tqdm(loader, desc="Eval", leave=False, disable=not sys.stdout.isatty())
        for smiles_batch, protein_batch, y in pbar:
            smiles_batch = _to_device(smiles_batch, self.device)
            protein_batch = _to_device(protein_batch, self.device)
            y = y.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits = self.model(smiles_batch, protein_batch).squeeze(1)
                loss_val = self.criterion(logits, y).item()
            total_loss += loss_val

            all_probs.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

        metrics = compute_classification_metrics(all_labels, all_probs)
        metrics["loss"] = total_loss / len(loader)
        return metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> nn.Module:
        """Train the model with early stopping and checkpointing.

        Returns:
            The model with the best validation loss weights loaded.
        """
        best_val_auc = 0.0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)
            val_loss = val_metrics["loss"]
            val_auc = val_metrics["auc"]

            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch:03d} | "
                f"time={epoch_time:.1f}s | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f}",
                flush=True
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                ckpt_path = self.checkpoint_dir / self.checkpoint_filename
                trainable_keys = {
                    n for n, p in self.model.named_parameters()
                    if p.requires_grad
                }
                buffer_keys = {n for n, _ in self.model.named_buffers()}
                save_keys = trainable_keys | buffer_keys
                state = {
                    # Strip `_orig_mod.` prefix from torch.compile()
                    k.removeprefix("_orig_mod."): v
                    for k, v in self.model.state_dict().items()
                    if k in save_keys
                }
                torch.save(state, ckpt_path)
                print(
                    f"  ✓ New best model (epoch {epoch}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}) saved to {ckpt_path}",
                    flush=True
                )

            if epoch - best_epoch >= self.patience:
                print(
                    f"Early stopping — no improvement for {self.patience} epochs"
                )
                break

        print(f"\nBest model: epoch {best_epoch}, val_loss={val_loss:.4f}, val_auc={best_val_auc:.4f}")

        # Load best checkpoint
        best_ckpt = self.checkpoint_dir / self.checkpoint_filename
        saved_state = torch.load(best_ckpt, weights_only=True)
        # If model is wrapped by torch.compile(), keys need _orig_mod. prefix
        if hasattr(self.model, "_orig_mod"):
            saved_state = {
                f"_orig_mod.{k}": v for k, v in saved_state.items()
            }
        self.model.load_state_dict(saved_state, strict=False)
        return self.model


def _to_device(obj, device: torch.device):
    """Move a tensor, PyG Batch, list, or dict of tensors to *device*.

    Uses ``non_blocking=True`` so CPU→GPU transfers can overlap with
    computation when ``pin_memory=True`` is set on the DataLoader.
    """
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(item, device) for item in obj)
    if hasattr(obj, "to"):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj
