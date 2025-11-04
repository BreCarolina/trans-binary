# -*- coding: utf-8 -*-
"""
prepare_dataset.py — MVP 2 (Normalização e Splits Temporais)

Lê o CSV transformado mais recente de data/processed/{PAR}_M{TF}_features_*x.csv,
gera splits temporais (train/val/test), normaliza as FEATURES (fit no TRAIN),
e salva X_train/X_val/X_test (+ y_*) em data/processed/ e um manifesto JSON em data/manifests/.

Uso (CLI):
    python scripts/prepare_dataset.py EURJPY m5
    python scripts/prepare_dataset.py EURUSD 300 0.7 0.15 0.15

Parâmetros:
    PAR           -> símbolo (ex.: EURUSD, EURJPY)
    TIMEFRAME     -> m1 | m5 | m15 | 60 | 300 | 900 (ou int)
    RATIOS op.    -> train, val, test (default 0.70 0.15 0.15) — devem somar 1.0

Saídas:
    data/processed/{PAR}_{Mtf}_X_train.csv
    data/processed/{PAR}_{Mtf}_y_train.csv
    data/processed/{PAR}_{Mtf}_X_val.csv
    data/processed/{PAR}_{Mtf}_y_val.csv
    data/processed/{PAR}_{Mtf}_X_test.csv
    data/processed/{PAR}_{Mtf}_y_test.csv
    data/manifests/{PAR}_{Mtf}_manifest.json
"""

from __future__ import annotations

import os
import sys
import json
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------
# Log sem buffer
# ---------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
except Exception:
    pass

# ---------------------------------------
# Caminhos
# ---------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MANI_DIR = ROOT / "data" / "manifests"
PROC_DIR.mkdir(parents=True, exist_ok=True)
MANI_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Utils
# ---------------------------------------
def parse_timeframe(val) -> int:
    if isinstance(val, int):
        return val
    s = str(val).strip().lower()
    mapa = {
        "m1": 60, "1m": 60, "60": 60,
        "m5": 300, "5m": 300, "300": 300,
        "m15": 900, "15m": 900, "900": 900,
    }
    if s in mapa:
        return mapa[s]
    if s.isdigit():
        return int(s)
    raise ValueError(f"TIMEFRAME inválido: {val}")

def tfmin_str(tf_s: int) -> str:
    return f"M{tf_s // 60}"

def achar_features_csv(par: str, tf_s: int) -> Optional[Path]:
    padrao = str(PROC_DIR / f"{par}_{tfmin_str(tf_s)}_features_*x.csv")
    cand = glob(padrao)
    if not cand:
        return None
    cand.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(cand[0])

def carregar_features(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["from"])
    df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)
    return df

def separar_colunas(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    base_cols = ["from", "abertura", "maxima", "minima", "fechamento", "volume"]
    label_cols = [c for c in df.columns if c.startswith("y_")]
    feat_cols = [c for c in df.columns if c not in base_cols + label_cols]
    return base_cols, feat_cols, label_cols

def splits_idx(n: int, ratios: Tuple[float,float,float]) -> Tuple[int,int]:
    r_train, r_val, r_test = ratios
    # limites (inteiros, respeitando ordem temporal)
    i_train_end = int(n * r_train)
    i_val_end = int(n * (r_train + r_val))
    # Garantias mínimas
    i_train_end = max(i_train_end, 1)
    i_val_end = max(i_val_end, i_train_end + 1)
    i_val_end = min(i_val_end, n - 1)
    return i_train_end, i_val_end

def fit_standard_scaler(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0).replace(0, 1.0)
    return {
        "mean": {k: float(mu[k]) for k in X.columns},
        "std":  {k: float(sigma[k]) for k in X.columns},
    }

def apply_standard_scaler(X: pd.DataFrame, scaler: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    mu = pd.Series(scaler["mean"])
    sigma = pd.Series(scaler["std"]).replace(0, 1.0)
    Xn = (X - mu) / sigma
    return Xn

# ---------------------------------------
# Pipeline
# ---------------------------------------
def preparar(par: str, tf_s: int, ratios: Tuple[float,float,float]) -> Dict[str, str]:
    src = achar_features_csv(par, tf_s)
    if not src:
        raise FileNotFoundError(f"Nenhum arquivo de features encontrado para {par} {tfmin_str(tf_s)} em {PROC_DIR}")
    print(f"[INFO] Usando features: {src}", flush=True)

    df = carregar_features(src)
    base_cols, feat_cols, label_cols = separar_colunas(df)
    if not feat_cols:
        raise RuntimeError("Nenhuma coluna de feature detectada.")
    if not label_cols:
        raise RuntimeError("Nenhuma coluna de rótulo (y_*) detectada.")

    n = len(df)
    i_train_end, i_val_end = splits_idx(n, ratios)
    print(f"[INFO] Tamanho total: {n} | train:0:{i_train_end}  val:{i_train_end}:{i_val_end}  test:{i_val_end}:{n}", flush=True)

    # Particiona temporalmente
    train_df = df.iloc[:i_train_end].copy()
    val_df   = df.iloc[i_train_end:i_val_end].copy()
    test_df  = df.iloc[i_val_end:].copy()

    # Separa X / y
    X_train, y_train = train_df[feat_cols], train_df[label_cols]
    X_val,   y_val   = val_df[feat_cols],   val_df[label_cols]
    X_test,  y_test  = test_df[feat_cols],  test_df[label_cols]

    # Normalização: fit no TRAIN, aplica em VAL/TEST
    scaler = fit_standard_scaler(X_train)
    X_train_n = apply_standard_scaler(X_train, scaler)
    X_val_n   = apply_standard_scaler(X_val, scaler)
    X_test_n  = apply_standard_scaler(X_test, scaler)

    # Caminhos de saída
    prefix = f"{par}_{tfmin_str(tf_s)}"
    out = {
        "X_train": str(PROC_DIR / f"{prefix}_X_train.csv"),
        "y_train": str(PROC_DIR / f"{prefix}_y_train.csv"),
        "X_val":   str(PROC_DIR / f"{prefix}_X_val.csv"),
        "y_val":   str(PROC_DIR / f"{prefix}_y_val.csv"),
        "X_test":  str(PROC_DIR / f"{prefix}_X_test.csv"),
        "y_test":  str(PROC_DIR / f"{prefix}_y_test.csv"),
        "manifest": str(MANI_DIR / f"{prefix}_manifest.json"),
        "source": str(src),
    }

    # Salva CSVs
    X_train_n.to_csv(out["X_train"], index=False)
    y_train.to_csv(out["y_train"], index=False)
    X_val_n.to_csv(out["X_val"], index=False)
    y_val.to_csv(out["y_val"], index=False)
    X_test_n.to_csv(out["X_test"], index=False)
    y_test.to_csv(out["y_test"], index=False)

    # Manifesto
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "par": par,
        "timeframe_s": tf_s,
        "tfmin": tfmin_str(tf_s),
        "source_csv": out["source"],
        "n_rows_source": int(n),
        "columns": {
            "base": base_cols,
            "features": feat_cols,
            "labels": label_cols,
        },
        "split": {
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "index_ranges": {
                "train": [0, i_train_end],
                "val":   [i_train_end, i_val_end],
                "test":  [i_val_end, n],
            },
            "date_ranges_utc": {
                "train": [train_df["from"].iloc[0].isoformat(), train_df["from"].iloc[-1].isoformat()],
                "val":   [val_df["from"].iloc[0].isoformat(),   val_df["from"].iloc[-1].isoformat()] if len(val_df) else None,
                "test":  [test_df["from"].iloc[0].isoformat(),  test_df["from"].iloc[-1].isoformat()] if len(test_df) else None,
            }
        },
        "scaler": {
            "type": "standard",
            "mean": {k: float(v) for k, v in pd.Series(scaler["mean"]).items()},
            "std":  {k: float(v) for k, v in pd.Series(scaler["std"]).items()},
            "epsilon_note": "std zeros substituídos por 1.0 para evitar divisão por zero",
            "fit_on": "train",
        },
        "artifacts": {
            "X_train": out["X_train"],
            "y_train": out["y_train"],
            "X_val":   out["X_val"],
            "y_val":   out["y_val"],
            "X_test":  out["X_test"],
            "y_test":  out["y_test"],
        }
    }
    with open(out["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[OK] Splits e normalização gerados.", flush=True)
    for k in ("X_train","y_train","X_val","y_val","X_test","y_test","manifest"):
        print(f" - {k}: {out[k]}", flush=True)

    return out

# ---------------------------------------
# CLI
# ---------------------------------------
def _print_uso():
    print("Uso: python scripts/prepare_dataset.py PAR TIMEFRAME [TRAIN VAL TEST]", flush=True)
    print("Ex.: python scripts/prepare_dataset.py EURJPY m5", flush=True)
    print("     python scripts/prepare_dataset.py EURUSD 300 0.7 0.2 0.1", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ("-h","--help"):
        _print_uso()
        sys.exit(1)

    par = sys.argv[1].upper()
    tf_val = sys.argv[2]
    ratios = (0.70, 0.15, 0.15)
    if len(sys.argv) >= 6:
        try:
            r_train = float(sys.argv[3])
            r_val   = float(sys.argv[4])
            r_test  = float(sys.argv[5])
            s = r_train + r_val + r_test
            if abs(s - 1.0) > 1e-9:
                raise ValueError("Ratios devem somar 1.0")
            ratios = (r_train, r_val, r_test)
        except Exception as e:
            print(f"[ERRO] Ratios inválidos: {e}", flush=True)
            _print_uso()
            sys.exit(2)

    try:
        tf_s = parse_timeframe(tf_val)
    except Exception as e:
        print(f"[ERRO] {e}", flush=True)
        _print_uso()
        sys.exit(2)

    try:
        preparar(par, tf_s, ratios)
        sys.exit(0)
    except Exception as e:
        print(f"[ERRO] Execução interrompida: {e}", flush=True)
        sys.exit(3)
