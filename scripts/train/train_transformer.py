#!/usr/bin/env python3
"""
Script de treinamento do Transformer (Fase-3) - versão com suporte a presets
Uso:
  python scripts/train/train_transformer.py --config src/config/models/transformer_baseline.yaml
"""
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict

import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.transformer.model import build_model_from_config

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def device_of(cfg):
    dev = cfg.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def load_numpy(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return np.load(path)

def make_dataloader(X_path, y_path, batch_size, shuffle):
    X = load_numpy(X_path)
    y = load_numpy(y_path)
    if X is None or y is None:
        return None
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def save_checkpoint(state, checkpoints_dir: str, epoch: int, max_checkpoints: int = 10):
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(checkpoints_dir, f"model_epoch_{epoch:03d}.pth")
    torch.save(state, fname)
    # cleanup old checkpoints
    ckpts = sorted(Path(checkpoints_dir).glob("model_epoch_*.pth"), key=os.path.getmtime, reverse=True)
    if len(ckpts) > max_checkpoints:
        for old in ckpts[max_checkpoints:]:
            try:
                old.unlink()
            except Exception:
                pass

def train_epoch(model, loader, optimizer, loss_fn, device):
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
        total_loss += loss.item() * Xb.size(0)
        n += Xb.size(0)
    return total_loss / max(1, n)

def eval_epoch(model, loader, loss_fn, device):
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
            correct += (preds == yb).sum().item()
            total_loss += loss.item() * Xb.size(0)
            n += Xb.size(0)
    if n == 0:
        return None, None
    return total_loss / n, correct / n

def infer_dataset_shape(path: str):
    """
    Return (seq_len, n_features) inferred from given numpy file.
    Uses memmap mode to avoid loading entire dataset in RAM.
    """
    if path is None:
        raise RuntimeError("Path para inferência é None")
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Arquivo para inferência não encontrado: {path}")
    arr = np.load(path, mmap_mode='r')
    if arr.ndim != 3:
        raise RuntimeError(f"Formato inválido em {path}. Esperado (N, seq_len, n_features), encontrado {arr.shape}")
    return int(arr.shape[1]), int(arr.shape[2])

def apply_preset_to_paths(cfg: Dict):
    """
    If cfg contains 'preset' and 'feature_presets', override cfg['paths'] accordingly.
    """
    preset = cfg.get("preset")
    presets = cfg.get("feature_presets")
    if preset and isinstance(presets, dict):
        if preset not in presets:
            raise KeyError(f"Preset '{preset}' não encontrado em cfg['feature_presets']")
        # override paths
        selected = presets[preset]
        if not isinstance(selected, dict):
            raise RuntimeError(f"Preset '{preset}' inválido (esperado mapping de paths)")
        cfg.setdefault("paths", {})
        for k, v in selected.items():
            cfg["paths"][k] = v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_of(cfg)
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 10))
    lr = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    checkpoints_dir = cfg.get("paths", {}).get("checkpoints_dir", "models/transformer/checkpoints")
    log_json = cfg.get("paths", {}).get("log_json", "logs/training/train_transformer_log.json")
    save_every = int(cfg.get("save_every_n_epochs", 1))
    max_ckpts = int(cfg.get("max_checkpoints", 10))

    # SUPPORT PRESET: override paths if preset present
    apply_preset_to_paths(cfg)

    # infer seq_len and n_features from train_X before building model
    train_X_path = cfg.get("paths", {}).get("train_X")
    if not train_X_path or not Path(train_X_path).exists():
        raise RuntimeError(f"Arquivo de treino não encontrado: {train_X_path}")

    inferred_seq_len, inferred_n_features = infer_dataset_shape(train_X_path)
    # Update cfg with inferred values (override YAML if present)
    cfg["seq_len"] = inferred_seq_len
    cfg["n_features"] = inferred_n_features

    # dataloaders (create after inference so shapes are known)
    train_loader = make_dataloader(
        cfg["paths"]["train_X"], cfg["paths"]["train_y"], batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        cfg["paths"].get("val_X"), cfg["paths"].get("val_y"), batch_size, shuffle=False
    )

    if train_loader is None:
        raise RuntimeError("Dados de treino não encontrados. Atualize paths no YAML e gere os .npy de treino.")

    # build model with inferred dims
    model = build_model_from_config(cfg)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    sch_cfg = cfg.get("scheduler", {})
    scheduler = None
    if sch_cfg.get("type") == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch_cfg.get("mode", "min"),
            factor=float(sch_cfg.get("factor", 0.5)),
            patience=int(sch_cfg.get("patience", 3)),
            min_lr=float(sch_cfg.get("min_lr", 1e-7))
        )

    # logging
    log = {
        "config": cfg,
        "runs": []
    }
    Path(os.path.dirname(log_json)).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)

        if scheduler is not None and val_loss is not None:
            scheduler.step(val_loss)

        epoch_time = time.time() - t0
        info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": epoch_time
        }
        print(json.dumps(info))
        log["runs"].append(info)

        # save checkpoint
        if epoch % save_every == 0:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg
            }
            save_checkpoint(state, checkpoints_dir, epoch, max_checkpoints=max_ckpts)

        # best
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            # save best
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "best": True
            }, checkpoints_dir, epoch, max_checkpoints=max_ckpts)

        # persist log each epoch
        with open(log_json, "w") as f:
            json.dump(log, f, indent=2, default=str)

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s, best_val_loss={best_val_loss}")

if __name__ == "__main__":
    main()
