#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparar_dados.py — Fase 2 (Normalização e Splits Temporais) 
Gera splits temporais (train/val/test), normaliza as FEATURES (fit no TRAIN),
permite escolher presets de features (baseline_basic | all | custom) via CLI
ou via arquivo de configuração YAML (--config). Salva X_*/y_* e manifesto.
"""
from __future__ import annotations

import os
import sys
import json
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timezone

import argparse
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None  # leitura de YAML é opcional; só usada se o usuário passar --config

# ---------------------------------------
# Paths
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
    i_train_end = int(n * r_train)
    i_val_end = int(n * (r_train + r_val))
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
# Feature selection presets
# ---------------------------------------
BASELINE_BASIC_KEEP = [
    "abertura", "maxima", "minima", "fechamento", "volume",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]

def select_features_df(df: pd.DataFrame, preset: str, features_list: Optional[List[str]]) -> pd.DataFrame:
    preset = (preset or "baseline_basic").lower()
    if preset in ("all", "none"):
        return df.copy()
    if preset == "baseline_basic":
        cols = [c for c in df.columns if c in BASELINE_BASIC_KEEP]
        if not cols:
            raise RuntimeError("Nenhuma coluna do baseline_basic encontrada no dataframe.")
        return df[cols].copy()
    if preset == "custom":
        if not features_list:
            raise RuntimeError("preset=custom mas features_list vazio.")
        cols = [c for c in features_list if c in df.columns]
        if not cols:
            raise RuntimeError("Nenhuma coluna válida encontrada para features_list custom.")
        return df[cols].copy()
    raise ValueError(f"Preset desconhecido: {preset}")

# ---------------------------------------
# Main pipeline
# ---------------------------------------
def preparar(par: str, tf_s: int, ratios: Tuple[float,float,float],
             preset: str = "baseline_basic", features_list: Optional[List[str]] = None,
             config_from_yaml: Optional[Dict] = None) -> Dict[str, str]:
    # If config_from_yaml provided and CLI args are defaults, allow YAML to influence preset/list
    if config_from_yaml:
        cfg_train = config_from_yaml.get("train", {}) if isinstance(config_from_yaml, dict) else {}
        if preset in (None, "", "yaml_default"):
            preset = cfg_train.get("feature_preset", preset)
        if not features_list:
            fl = cfg_train.get("features_list", None)
            if isinstance(fl, list) and fl:
                features_list = fl

    src = achar_features_csv(par, tf_s)
    if not src:
        raise FileNotFoundError(f"Nenhum arquivo de features encontrado para {par} {tfmin_str(tf_s)} em {PROC_DIR}")
    print(f"[INFO] Usando features (bruto): {src}", flush=True)

    df = carregar_features(src)
    base_cols, feat_cols, label_cols = separar_colunas(df)
    if not feat_cols:
        raise RuntimeError("Nenhuma coluna de feature detectada no CSV de entrada.")
    if not label_cols:
        raise RuntimeError("Nenhuma coluna de rótulo (y_*) detectada no CSV de entrada.")

    # Seleciona features conforme preset/custom — aplicamos apenas sobre as feature columns
    selected_feat_df = select_features_df(df[feat_cols], preset, features_list)

    # Reconstrói dataframe com base + selected_features + labels
    df_proc = pd.concat([df[base_cols], selected_feat_df, df[label_cols]], axis=1)

    # Índices e splits
    n = len(df_proc)
    if n < 3:
        raise RuntimeError("Número de linhas insuficiente para gerar splits.")
    i_train_end, i_val_end = splits_idx(n, ratios)
    print(f"[INFO] Tamanho total: {n} | train:0:{i_train_end}  val:{i_train_end}:{i_val_end}  test:{i_val_end}:{n}", flush=True)

    train_df = df_proc.iloc[:i_train_end].copy()
    val_df   = df_proc.iloc[i_train_end:i_val_end].copy()
    test_df  = df_proc.iloc[i_val_end:].copy()

    feat_cols_sel = list(selected_feat_df.columns)
    # Separa X / y
    X_train, y_train = train_df[feat_cols_sel], train_df[label_cols]
    X_val,   y_val   = val_df[feat_cols_sel],   val_df[label_cols]
    X_test,  y_test  = test_df[feat_cols_sel],  test_df[label_cols]

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
            "features_selected": feat_cols_sel,
            "labels": label_cols,
        },
        "feature_selection": {
            "preset": preset,
            "features_list_provided": features_list if features_list else None,
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
    print("Uso: python scripts/preparar_dados.py PAR TIMEFRAME [--train-val-test 0.7 0.15 0.15] [--preset baseline_basic|all|custom] [--features col1,col2,...] [--config cfg.yaml]", flush=True)
    print("Ex.: python scripts/preparar_dados.py EURJPY m5", flush=True)
    print("     python scripts/preparar_dados.py EURJPY m5 --preset all", flush=True)
    print("     python scripts/preparar_dados.py EURJPY m5 --preset custom --features abertura,fechamento,volume", flush=True)
    print("     python scripts/preparar_dados.py EURJPY m5 --config src/config/eurjpy_m5.yaml", flush=True)

def _load_yaml(path: str) -> dict:
    if not yaml:
        raise RuntimeError("yaml não disponível (pip install pyyaml) mas --config foi passado.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="preparar_dados.py")
    ap.add_argument("par", type=str, help="Símbolo (ex.: EURJPY)")
    ap.add_argument("timeframe", type=str, help="Timeframe (ex.: m5 ou 300)")
    ap.add_argument("--train-val-test", nargs=3, type=float, metavar=("TRAIN","VAL","TEST"),
                    help="Ratios 3 valores que somam 1.0 (default 0.7 0.15 0.15)")
    ap.add_argument("--preset", type=str, default=None,
                    help="Preset de features: baseline_basic | all | custom (padrão: do YAML ou baseline_basic)")
    ap.add_argument("--features", type=str, default=None,
                    help="Lista CSV de colunas quando --preset custom, ex: abertura,fechamento,volume")
    ap.add_argument("--config", type=str, default=None,
                    help="Caminho opcional para um YAML de configuração (se fornecido pode definir feature_preset e features_list)")
    args = ap.parse_args()

    par = args.par.upper()
    try:
        tf_s = parse_timeframe(args.timeframe)
    except Exception as e:
        print(f"[ERRO] TIMEFRAME inválido: {e}", flush=True)
        _print_uso()
        sys.exit(2)

    ratios = (0.70, 0.15, 0.15)
    if args.train_val_test:
        try:
            r_train, r_val, r_test = args.train_val_test
            s = r_train + r_val + r_test
            if abs(s - 1.0) > 1e-9:
                raise ValueError("Ratios devem somar 1.0")
            ratios = (r_train, r_val, r_test)
        except Exception as e:
            print(f"[ERRO] Ratios inválidos: {e}", flush=True)
            _print_uso()
            sys.exit(2)

    preset = args.preset if args.preset else "yaml_default"
    features_list = None
    if args.features:
        features_list = [c.strip() for c in args.features.split(",") if c.strip()]

    cfg = None
    if args.config:
        try:
            cfg = _load_yaml(args.config)
        except Exception as e:
            print(f"[ERRO] Falha ao ler config YAML: {e}", flush=True)
            sys.exit(2)

    try:
        preparar(par, tf_s, ratios, preset=preset, features_list=features_list, config_from_yaml=cfg)
        sys.exit(0)
    except Exception as e:
        print(f"[ERRO] Execução interrompida: {e}", flush=True)
        sys.exit(3)
