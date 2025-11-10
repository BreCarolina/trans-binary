# -*- coding: utf-8 -*-
"""
treinar_transformer.py — Treinamento do Transformer (ajustado ao novo pipeline)
Uso:
  PYTHONPATH="$PWD" python scripts/treinar_transformer.py src/config/eurjpy_m5.yaml
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from datetime import datetime

# importa model e utilidades (ajuste nomes se necessário)
from src.model.transformer import TimeSeriesTransformer
from src.model.sequenciar_dados import make_sequences, build_target_vector, load_config as sd_load_config

# -------------------------
# Utils
# -------------------------
def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def choose_device(device_cfg: str) -> torch.device:
    if device_cfg and device_cfg.lower() in ("cpu", "cuda"):
        if device_cfg.lower() == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA solicitado, mas não disponível. Usando CPU.", flush=True)
            return torch.device("cpu")
        return torch.device(device_cfg.lower())
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["from"])

def select_feature_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """
    Presets: baseline_basic -> keeps OHLCV + time features (hour_sin/hour_cos/dow_sin/dow_cos)
    If preset is None or 'all' -> returns df unchanged.
    """
    if not preset or preset in ("all", "none"):
        return df
    if preset == "baseline_basic":
        keep = ["abertura", "maxima", "minima", "fechamento", "volume",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        # keep only those that exist, keep order as in df
        cols = [c for c in df.columns if c in keep]
        return df[cols].copy()
    raise ValueError(f"Preset desconhecido: {preset}")

def df_to_tensor_dataset(X: np.ndarray, y: np.ndarray, device: torch.device) -> TensorDataset:
    Xt = torch.from_numpy(X).to(torch.float32)
    yt = torch.from_numpy(y).to(torch.long)
    return TensorDataset(Xt, yt)

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

# -------------------------
# Pipeline de treinamento
# -------------------------
def prepare_sequences_from_splits(cfg: Dict, feature_preset: str = "baseline_basic") -> Dict[str, np.ndarray]:
    """
    Carrega X_train/val/test CSVs, aplica seleção de preset (se solicitado),
    gera vetores alvo (usando build_target_vector) e cria sequências com make_sequences.
    Retorna dicionário com arrays prontos para DataLoader.
    """
    data = cfg["data"]
    seq_cfg = cfg["sequence"]
    seq_len = int(seq_cfg["seq_len"])
    horizon = int(seq_cfg.get("horizon", 1))
    target = seq_cfg.get("target", "direction")
    target_map = seq_cfg.get("target_map", {})

    # Lê CSVs
    X_tr = read_csv(data["X_train"])
    y_tr = read_csv(data["y_train"])
    X_va = read_csv(data["X_val"])
    y_va = read_csv(data["y_val"])
    X_te = read_csv(data["X_test"])
    y_te = read_csv(data["y_test"])

    # Aplica preset (mantendo apenas colunas selecionadas)
    X_tr_sel = select_feature_preset(X_tr, feature_preset)
    X_va_sel = select_feature_preset(X_va, feature_preset)
    X_te_sel = select_feature_preset(X_te, feature_preset)

    # Construir vetores alvo (note que sequenciar_dados espera dataframes y_* com colunas y_up/y_down/y_noop)
    y_tr_vec = build_target_vector(y_tr, target, target_map)
    y_va_vec = build_target_vector(y_va, target, target_map)
    y_te_vec = build_target_vector(y_te, target, target_map)

    # Se target == "direction" our y values are in {-1,0,1}; map to indices 0,1,2
    if target == "direction":
        y_tr_vec = (y_tr_vec + 1).astype(np.int64)
        y_va_vec = (y_va_vec + 1).astype(np.int64)
        y_te_vec = (y_te_vec + 1).astype(np.int64)

    # make_sequences espera DataFrame X e vetor y
    Xtr_seq, ytr = make_sequences(X_tr_sel, y_tr_vec, seq_len, horizon)
    Xva_seq, yva = make_sequences(X_va_sel, y_va_vec, seq_len, horizon)
    Xte_seq, yte = make_sequences(X_te_sel, y_te_vec, seq_len, horizon)

    return {
        "X_train": Xtr_seq, "y_train": ytr,
        "X_val": Xva_seq,   "y_val": yva,
        "X_test": Xte_seq,  "y_test": yte,
        "feature_names": list(X_tr_sel.columns),
        "seq_len": seq_len,
        "horizon": horizon,
    }

def train_loop(model, optim, loss_fn, dl):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for Xb, yb in dl:
        Xb = Xb.to(next(model.parameters()).device)
        yb = yb.to(next(model.parameters()).device)
        optim.zero_grad()
        logits = model(Xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        total_loss += float(loss.item()) * Xb.size(0)
        total_acc += accuracy_from_logits(logits.detach().cpu(), yb.detach().cpu()) * Xb.size(0)
        n += Xb.size(0)
    return total_loss / n, total_acc / n

def eval_loop(model, loss_fn, dl):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    with torch.no_grad():
        for Xb, yb in dl:
            Xb = Xb.to(next(model.parameters()).device)
            yb = yb.to(next(model.parameters()).device)
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * Xb.size(0)
            total_acc += accuracy_from_logits(logits.detach().cpu(), yb.detach().cpu()) * Xb.size(0)
            n += Xb.size(0)
    return total_loss / n, total_acc / n

def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = load_yaml(cfg_path)

    # device
    device = choose_device(cfg.get("train", {}).get("device", "auto"))
    print(f"[INFO] Device: {device}", flush=True)

    # prepare sequences (feature preset default: baseline_basic)
    feature_preset = cfg.get("train", {}).get("feature_preset", "baseline_basic")
    seqpack = prepare_sequences_from_splits(cfg, feature_preset=feature_preset)

    Xtr, ytr = seqpack["X_train"], seqpack["y_train"]
    Xva, yva = seqpack["X_val"],   seqpack["y_val"]
    Xte, yte = seqpack["X_test"],  seqpack["y_test"]

    n_features = Xtr.shape[2]
    seq_len = Xtr.shape[1]
    n_classes = int(np.unique(ytr).size)

    print("[SEQ] Shapes:")
    print(" - X_train:", Xtr.shape, "| y_train:", ytr.shape)
    print(" - X_val:  ", Xva.shape, "| y_val:  ", yva.shape)
    print(" - X_test: ", Xte.shape, "| y_test: ", yte.shape)
    print(f" - seq_len: {seq_len} | n_features: {n_features} | n_classes: {n_classes}", flush=True)

    # Model config
    mcfg = cfg.get("model", {})
    model = TimeSeriesTransformer(
        n_features=n_features,
        seq_len=seq_len,
        d_model=int(mcfg.get("d_model", 128)),
        nhead=int(mcfg.get("nhead", 8)),
        num_layers=int(mcfg.get("num_layers", 4)),
        dim_feedforward=int(mcfg.get("dim_feedforward", 256)),
        dropout=float(mcfg.get("dropout", 0.1)),
        activation=str(mcfg.get("activation", "gelu")),
        num_classes=n_classes,
    ).to(device)

    # training params
    tcfg = cfg.get("train", {})
    batch_size = int(tcfg.get("batch_size", 256))
    lr = float(tcfg.get("lr", 1.5e-4))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    max_epochs = int(tcfg.get("max_epochs", 30))
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    # dataloaders
    train_ds = df_to_tensor_dataset(Xtr, ytr, device)
    val_ds   = df_to_tensor_dataset(Xva, yva, device)
    test_ds  = df_to_tensor_dataset(Xte, yte, device)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # paths
    paths = cfg.get("paths", {})
    ckpt_dir = Path(paths.get("checkpoints_dir", "models/transformer"))
    log_dir = Path(paths.get("logs_dir", "logs/training"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_ckpt = None
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_loop(model, optim, loss_fn, train_dl)

        # optional grad clip
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        val_loss, val_acc = eval_loop(model, loss_fn, val_dl)

        now = datetime.utcnow().isoformat()
        print(f"[E{epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

        history.append({
            "epoch": epoch,
            "ts": now,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # salvar melhor
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / f"PAIR_TF_transformer_best_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "cfg": cfg,
            }, ckpt_path)
            best_ckpt = str(ckpt_path)
            print(f"[OK] Novo melhor modelo salvo em: {ckpt_path}", flush=True)

    # avaliação final em test
    test_loss, test_acc = eval_loop(model, loss_fn, test_dl)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}", flush=True)

    # salva log
    log_path = log_dir / f"train_transformer_log_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
    meta = {
        "config": str(cfg_path),
        "device": str(device),
        "best_ckpt": best_ckpt,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
        "feature_preset": feature_preset,
        "feature_names": seqpack["feature_names"],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Log salvo em: {log_path}", flush=True)
    print("[OK] Treinamento finalizado.", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/treinar_transformer.py path/to/config.yaml", flush=True)
        sys.exit(1)
    main(sys.argv[1])
