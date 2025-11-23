#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treinamento do Transformer (Fase-3) com suporte a presets (baseline | all | custom).

Uso:
    python3 src/transformer/treinamento.py --config src/transformer/config/transformer_baseline.yaml
"""
from __future__ import annotations

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import model builder (ajuste conforme layout src/transformer/model.py)
from src.transformer.model import build_model_from_config


# -------------------- utilidades --------------------
def load_config(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML não encontrado: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def device_of(cfg: Dict) -> torch.device:
    dev = str(cfg.get("device", "auto")).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def apply_preset_to_paths(cfg: Dict) -> None:
    """
    Se 'preset' e 'feature_presets' estiverem na cfg, sobrescreve cfg['paths']
    com os caminhos definidos no preset (train_X, train_y, val_X, val_y, test_X, test_y).
    """
    preset = cfg.get("preset")
    presets = cfg.get("feature_presets")
    if preset and isinstance(presets, dict):
        if preset not in presets:
            raise KeyError(f"Preset '{preset}' não encontrado em cfg['feature_presets']")
        selected = presets[preset]
        if not isinstance(selected, dict):
            raise RuntimeError(f"Preset '{preset}' inválido (esperado mapping de paths)")
        cfg.setdefault("paths", {})
        for k, v in selected.items():
            cfg["paths"][k] = v


def infer_dataset_shape(path: str):
    """
    Retorna (seq_len, n_features) inferidos a partir de um arquivo numpy (N, seq_len, n_features).
    Usa mmap para não carregar tudo na memória.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo para inferência não encontrado: {path}")
    arr = np.load(str(p), mmap_mode="r")
    if arr.ndim != 3:
        raise RuntimeError(f"Formato inválido em {path}. Esperado (N, seq_len, n_features), encontrado {arr.shape}")
    return int(arr.shape[1]), int(arr.shape[2])


def load_numpy_strict(path: Optional[str], allow_none: bool = False):
    if not path:
        if allow_none:
            return None
        raise FileNotFoundError("Caminho .npy não informado.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo .npy não encontrado: {path}")
    return np.load(str(p))


def make_dataloader_from_numpy(X_path: str, y_path: str, batch_size: int, shuffle: bool):
    X = load_numpy_strict(X_path)
    y = load_numpy_strict(y_path)
    if X.ndim != 3:
        raise RuntimeError(f"X deve ter shape (N, seq_len, n_features). Encontrado: {X.shape}")
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def save_checkpoint(state: Dict, checkpoints_dir: str, epoch: int, max_checkpoints: int = 10):
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(checkpoints_dir) / f"model_epoch_{epoch:03d}.pth"
    torch.save(state, str(fname))
    # cleanup
    ckpts = sorted(Path(checkpoints_dir).glob("model_epoch_*.pth"), key=os.path.getmtime, reverse=True)
    if len(ckpts) > max_checkpoints:
        for old in ckpts[max_checkpoints:]:
            try:
                old.unlink()
            except Exception:
                pass


# -------------------- treino / avaliação --------------------
def train_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn, device: torch.device):
    model.train()
    total_loss = 0.0
    n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * Xb.size(0)
        n += Xb.size(0)
    return total_loss / max(1, n)


def eval_epoch(model: nn.Module, loader: Optional[DataLoader], loss_fn, device: torch.device):
    if loader is None:
        return None, None
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total_loss += float(loss.item()) * Xb.size(0)
            n += Xb.size(0)
    if n == 0:
        return None, None
    return total_loss / n, correct / n


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(prog="train_transformer.py")
    ap.add_argument("--config", type=str, required=True, help="Caminho para YAML de configuração")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # apply preset if present (overrides paths)
    apply_preset_to_paths(cfg)

    device = device_of(cfg)
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 10))
    lr = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    checkpoints_dir = cfg.get("paths", {}).get("checkpoints_dir", cfg.get("paths", {}).get("checkpoints", "models/transformer/checkpoints"))
    log_json = cfg.get("paths", {}).get("log_json", "logs/training/train_transformer_log.json")
    save_every = int(cfg.get("save_every_n_epochs", 1))
    max_ckpts = int(cfg.get("max_checkpoints", 10))

    # infer seq_len & n_features from train_X (preset should have set these paths)
    train_X_path = cfg.get("paths", {}).get("train_X")
    train_y_path = cfg.get("paths", {}).get("train_y")
    if not train_X_path or not train_y_path:
        raise RuntimeError("train_X / train_y não configurados em cfg['paths'] (ver preset ou YAML).")

    seq_len, n_features = infer_dataset_shape(train_X_path)
    # override cfg with inferred
    cfg["seq_len"] = seq_len
    cfg["n_features"] = n_features

    # build dataloaders
    train_loader = make_dataloader_from_numpy(train_X_path, train_y_path, batch_size=batch_size, shuffle=True)
    val_loader = None
    val_X = cfg.get("paths", {}).get("val_X")
    val_y = cfg.get("paths", {}).get("val_y")
    if val_X and val_y:
        try:
            val_loader = make_dataloader_from_numpy(val_X, val_y, batch_size=batch_size, shuffle=False)
        except Exception as e:
            print(f"[WARN] Falha ao criar val_loader: {e}")

    # build model from cfg (will pick seq_len/n_features from cfg)
    model = build_model_from_config(cfg)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler support (ReduceLROnPlateau)
    scheduler = None
    sch_cfg = cfg.get("scheduler", {}) or {}
    if sch_cfg.get("type") == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch_cfg.get("mode", "min"),
            factor=float(sch_cfg.get("factor", 0.5)),
            patience=int(sch_cfg.get("patience", 3)),
            min_lr=float(sch_cfg.get("min_lr", 1e-7)),
        )

    # prepare logging
    Path(log_json).parent.mkdir(parents=True, exist_ok=True)
    log = {"config": cfg, "runs": []}

    best_val_loss = float("inf")
    start_all = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)

        # scheduler step
        if scheduler is not None and val_loss is not None:
            scheduler.step(val_loss)

        epoch_time = time.time() - t0
        info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_s": epoch_time,
        }
        print(json.dumps(info))
        log["runs"].append(info)

        # checkpoint periodic
        if save_every > 0 and (epoch % save_every == 0):
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }
            save_checkpoint(state, checkpoints_dir, epoch, max_checkpoints=max_ckpts)

        # best model save
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "best": True,
            }, checkpoints_dir, epoch, max_checkpoints=max_ckpts)

        # persist log
        with open(log_json, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)

    total_time = time.time() - start_all
    print(f"Training finished in {total_time:.1f}s, best_val_loss={best_val_loss}")

    # final test evaluation (if paths configured in preset/YAML)
    print("\n[TEST] Avaliando no conjunto de teste...")
    test_X = cfg.get("paths", {}).get("test_X")
    test_y = cfg.get("paths", {}).get("test_y")
    if not test_X or not test_y:
        print("[WARN] test_X/test_y não configurados; pulando avaliação final.")
        return

    # load test arrays (in memory; cuidado com RAM se arquivo grande)
    X_test = load_numpy_strict(test_X)
    y_test = load_numpy_strict(test_y)
    if X_test.ndim != 3:
        print(f"[WARN] test_X shape inesperado: {X_test.shape}; pulando teste.")
        return

    model.eval()
    correct = 0
    total = 0
    bs = batch_size
    with torch.no_grad():
        for i in range(0, len(X_test), bs):
            batch_X = torch.from_numpy(X_test[i : i + bs]).float().to(device)
            batch_y = torch.from_numpy(y_test[i : i + bs]).long().to(device)
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            correct += int((preds == batch_y).sum().item())
            total += batch_y.size(0)

    test_acc = correct / total if total > 0 else 0.0
    print(f"[TEST] Acurácia final: {test_acc:.4f}  ({correct}/{total})")


if __name__ == "__main__":
    main()
