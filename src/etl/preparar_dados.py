#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparar_dados.py — Fase 2 (Normalização, Splits Temporais e Sequências)

- Lê CSV de features já transformadas (fase anterior)
- Seleciona features por preset (baseline | all | custom)
- Gera splits temporais (train/val/test)
- Normaliza features (fit no TRAIN, aplica em VAL/TEST)
- Cria sequências (janelas) para o modelo Transformer
- Salva CSVs normalizados, arrays .npy janelados e manifesto JSON
"""
from __future__ import annotations

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_BASE = ROOT / 'data' / 'processed'
PROCESSED_BASE.mkdir(parents=True, exist_ok=True)

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
    yaml = None

# -------------------- Paths / Consts --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
SRC_DIR = PROJECT_ROOT / "src"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROCESSED_BASE
MANI_DIR = PROJECT_ROOT / "data" / "manifests"
PROC_DIR.mkdir(parents=True, exist_ok=True)
MANI_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN_DEFAULT = 60

# Ensure src importable when running via python3 src/...
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -------------------- Helpers --------------------
def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML não disponível; instale 'pyyaml' para usar --config.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML não encontrado: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def parse_timeframe(val) -> int:
    if isinstance(val, int):
        return val
    s = str(val).strip().lower()
    mapa = {"m1": 60, "1m": 60, "60": 60,
            "m5": 300, "5m": 300, "300": 300,
            "m15": 900, "15m": 900, "900": 900}
    if s in mapa:
        return mapa[s]
    if s.isdigit():
        return int(s)
    raise ValueError(f"TIMEFRAME inválido: {val}")

def tfmin_str(tf_s: int) -> str:
    return f"M{int(tf_s) // 60}"

def achar_features_csv(par: str, tf_s: int) -> Optional[Path]:
    padrao = str(PROC_DIR / f"{par}_{tfmin_str(tf_s)}_*x.csv")
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
    label_cols = [c for c in df.columns if c.startswith("y_") or c == "target_bin" or c == "target_ternary"]
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

# -------------------- Presets --------------------
BASELINE_KEEP = [
    "abertura", "maxima", "minima", "fechamento", "volume",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]

def select_features_df(df: pd.DataFrame, preset: str, features_list: Optional[List[str]]) -> pd.DataFrame:
    p = (preset or "baseline").lower()
    if p in ("all", "full"):
        return df.copy()
    if p == "baseline":
        cols = [c for c in df.columns if c in BASELINE_KEEP]
        if not cols:
            raise RuntimeError("Nenhuma coluna do baseline encontrada no dataframe.")
        return df[cols].copy()
    if p == "custom":
        if not features_list:
            raise RuntimeError("preset=custom mas features_list vazio.")
        cols = [c for c in features_list if c in df.columns]
        if not cols:
            raise RuntimeError("Nenhuma coluna válida encontrada para features_list custom.")
        return df[cols].copy()
    raise ValueError(f"Preset desconhecido: {preset}")

# -------------------- Sequências --------------------
def build_sequences_numpy(X_df: pd.DataFrame, y_arr, seq_len: int):
    X_np = X_df.to_numpy(dtype="float32")
    y_np = np.asarray(y_arr, dtype="int64")
    n_samples, n_feat = X_np.shape
    if n_samples < seq_len:
        raise RuntimeError(f"Número de linhas ({n_samples}) menor que seq_len={seq_len}.")
    X_seq = []
    y_seq = []
    for i in range(n_samples - seq_len + 1):
        X_seq.append(X_np[i:i+seq_len, :])
        y_seq.append(y_np[i+seq_len-1])
    return np.stack(X_seq, axis=0), np.asarray(y_seq, dtype="int64")

# -------------------- Pipeline --------------------
def preparar(par: str,
             tf_s: int,
             ratios: tuple,
             preset: str = "baseline",
             features_list: Optional[list] = None,
             config_from_yaml: Optional[dict] = None,
             seq_len: int = SEQ_LEN_DEFAULT) -> dict:
    # Allow YAML override
    if config_from_yaml and isinstance(config_from_yaml, dict):
        cfg_train = config_from_yaml.get("train", {})
        if preset in (None, "", "yaml_default"):
            preset = cfg_train.get("feature_preset", preset) or "baseline"
        if not features_list:
            fl = cfg_train.get("features_list")
            if isinstance(fl, list) and fl:
                features_list = fl
        if "seq_len" in cfg_train and (not seq_len or seq_len == SEQ_LEN_DEFAULT):
            try:
                seq_len = int(cfg_train["seq_len"])
            except Exception:
                pass

    src = achar_features_csv(par, tf_s)
    if not src:
        raise FileNotFoundError(f"Nenhum arquivo de features encontrado para {par} {tfmin_str(tf_s)} em {PROC_DIR}")
    print(f"[INFO] Usando features (bruto): {src}", flush=True)

    df = carregar_features(src)
    base_cols, feat_cols, label_cols = separar_colunas(df)
    if not feat_cols:
        raise RuntimeError("Nenhuma coluna de feature detectada no CSV de entrada.")

    # select features
    selected_feat_df = select_features_df(df[feat_cols], preset, features_list)
    feat_cols_sel = list(selected_feat_df.columns)

    # validate label column presence
    if "target_bin" not in df.columns and not any(c in df.columns for c in ("y_up","y_down")):
        raise RuntimeError("Coluna target_bin ou y_up/y_down não encontrada no CSV de entrada.")

    # rebuild processed df: base + selected_features + labels
    extra_label_cols = [c for c in ("y_up","y_down","y_noop","target_ternary") if c in df.columns]
    cols_label = ["target_bin"] + extra_label_cols
    cols_label = [c for c in cols_label if c in df.columns]
    df_proc = pd.concat([df[base_cols], selected_feat_df, df[cols_label]], axis=1)

    # splits
    n = len(df_proc)
    if n < 3:
        raise RuntimeError("Número de linhas insuficiente para gerar splits.")
    i_train_end, i_val_end = splits_idx(n, ratios)
    print(f"[INFO] Tamanho total: {n} | train:0:{i_train_end}  val:{i_train_end}:{i_val_end}  test:{i_val_end}:{n}", flush=True)

    train_df = df_proc.iloc[:i_train_end].copy()
    val_df   = df_proc.iloc[i_train_end:i_val_end].copy()
    test_df  = df_proc.iloc[i_val_end:].copy()

    # choose label strategy
    if "y_up" in df_proc.columns and "y_down" in df_proc.columns:
        label_col = "y_up"
        mask_train = (train_df["y_up"] == 1) | (train_df["y_down"] == 1)
        mask_val   = (val_df["y_up"] == 1) | (val_df["y_down"] == 1)
        mask_test  = (test_df["y_up"] == 1) | (test_df["y_down"] == 1)

        if not mask_train.any():
            raise RuntimeError("Nenhum exemplo forte (y_up/y_down) encontrado no treino.")
        if not mask_val.any():
            raise RuntimeError("Nenhum exemplo forte (y_up/y_down) encontrado na validação.")
        if not mask_test.any():
            raise RuntimeError("Nenhum exemplo forte (y_up/y_down) encontrado no teste.")

        X_train = train_df.loc[mask_train, feat_cols_sel]
        y_train = train_df.loc[mask_train, label_col].astype("int64")

        X_val   = val_df.loc[mask_val, feat_cols_sel]
        y_val   = val_df.loc[mask_val, label_col].astype("int64")

        X_test  = test_df.loc[mask_test, feat_cols_sel]
        y_test  = test_df.loc[mask_test, label_col].astype("int64")
        label_target = label_col
    else:
        label_target = "target_bin"
        X_train = train_df[feat_cols_sel]
        X_val   = val_df[feat_cols_sel]
        X_test  = test_df[feat_cols_sel]
        y_train = train_df[label_target].astype("int64")
        y_val   = val_df[label_target].astype("int64")
        y_test  = test_df[label_target].astype("int64")

    # normalization (fit on train)
    scaler = fit_standard_scaler(X_train)
    X_train_n = apply_standard_scaler(X_train, scaler)
    X_val_n   = apply_standard_scaler(X_val, scaler)
    X_test_n  = apply_standard_scaler(X_test, scaler)

    # CSV artifact paths
    prefix = f"{par}_{tfmin_str(tf_s)}"
    preset_name = (preset or "baseline").lower()
    preset_dir = PROC_DIR / preset_name
    preset_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "X_train": str(preset_dir / f"{prefix}_{preset_name}_X_train.csv"),
        "y_train": str(preset_dir / f"{prefix}_{preset_name}_y_train.csv"),
        "X_val":   str(preset_dir / f"{prefix}_{preset_name}_X_val.csv"),
        "y_val":   str(preset_dir / f"{prefix}_{preset_name}_y_val.csv"),
        "X_test":  str(preset_dir / f"{prefix}_{preset_name}_X_test.csv"),
        "y_test":  str(preset_dir / f"{prefix}_{preset_name}_y_test.csv"),
        "manifest": str(MANI_DIR / f"{prefix}_{preset_name}_manifest.json"),
        "source": str(src),
    }

    # save CSVs
    X_train_n.to_csv(out["X_train"], index=False)
    pd.DataFrame(y_train).to_csv(out["y_train"], index=False)
    X_val_n.to_csv(out["X_val"], index=False)
    pd.DataFrame(y_val).to_csv(out["y_val"], index=False)
    X_test_n.to_csv(out["X_test"], index=False)
    pd.DataFrame(y_test).to_csv(out["y_test"], index=False)

    # build sequences .npy
    if seq_len is None or seq_len <= 0:
        raise ValueError("seq_len deve ser um inteiro positivo.")

    X_train_seq, y_train_seq = build_sequences_numpy(X_train_n, y_train, seq_len)
    X_val_seq,   y_val_seq   = build_sequences_numpy(X_val_n,   y_val,   seq_len)
    X_test_seq,  y_test_seq  = build_sequences_numpy(X_test_n,  y_test,  seq_len)

    out_npy = {
        "X_train_npy": str(preset_dir / f"{prefix}_{preset_name}_X_train.npy"),
        "y_train_npy": str(preset_dir / f"{prefix}_{preset_name}_y_train.npy"),
        "X_val_npy":   str(preset_dir / f"{prefix}_{preset_name}_X_val.npy"),
        "y_val_npy":   str(preset_dir / f"{prefix}_{preset_name}_y_val.npy"),
        "X_test_npy":  str(preset_dir / f"{prefix}_{preset_name}_X_test.npy"),
        "y_test_npy":  str(preset_dir / f"{prefix}_{preset_name}_y_test.npy"),
    }


    # --- Forçar saída para data/processed/baseline_basic (padronizado) ---

    BASELINE_BASIC_DIR = PROCESSED_BASE / 'baseline_basic'

    BASELINE_BASIC_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = {

        'X_train_csv': str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_X_train.csv"),

        'y_train_csv': str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_y_train.csv"),

        'X_val_csv':   str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_X_val.csv"),

        'y_val_csv':   str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_y_val.csv"),

        'X_test_csv':  str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_X_test.csv"),

        'y_test_csv':  str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_y_test.csv"),

    }


    out_npy = {

        'X_train_npy': str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_X_train.npy"),

        'y_train_npy': str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_y_train.npy"),

        'X_val_npy':   str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_X_val.npy"),

        'y_val_npy':   str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_y_val.npy"),

        'X_test_npy':  str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_X_test.npy"),

        'y_test_npy':  str(BASELINE_BASIC_DIR / f"{par}_{tfmin_str(tf_s)}_baseline_basic_y_test.npy"),

    }


    np.save(out_npy["X_train_npy"], X_train_seq)
    np.save(out_npy["y_train_npy"], y_train_seq)
    np.save(out_npy["X_val_npy"],   X_val_seq)
    np.save(out_npy["y_val_npy"],   y_val_seq)
    np.save(out_npy["X_test_npy"],  X_test_seq)
    np.save(out_npy["y_test_npy"],  y_test_seq)

    # manifest
    manifest = {
        "par": par,
        "timeframe_s": int(tf_s),
        "tfmin": tfmin_str(tf_s),
        "created_from": str(src),
        "n_rows_source": int(n),
        "feature_preset": preset_name,
        "features_used": feat_cols_sel,
        "label_target": label_target,
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "index_ranges": {
            "train": [0, i_train_end],
            "val":   [i_train_end, i_val_end],
            "test":  [i_val_end, n],
        },
        "seq_len": int(seq_len),
        "n_features": len(feat_cols_sel),
        "scaler": scaler,
        "csv_artifacts": out,
        "npy_artifacts": out_npy,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(out["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[OK] Splits, normalização e sequências gerados.", flush=True)
    for k in ("X_train","y_train","X_val","y_val","X_test","y_test","manifest"):
        print(f" - {k}: {out[k]}", flush=True)
    for k in ("X_train_npy","y_train_npy","X_val_npy","y_val_npy","X_test_npy","y_test_npy"):
        print(f" - {k}: {out_npy[k]}", flush=True)

    return {**out, **out_npy}

# -------------------- CLI --------------------
def _print_uso():
    print("Uso: python3 src/etl/preparar_dados.py PAR TIMEFRAME [--preset baseline|all|custom] [--features a,b,c]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="preparar_dados.py")
    ap.add_argument("par", type=str, help="Símbolo (ex.: EURJPY)")
    ap.add_argument("timeframe", type=str, help="Timeframe (ex.: m5 ou 300)")
    ap.add_argument("--train-val-test", nargs=3, type=float, metavar=("TRAIN","VAL","TEST"),
                    help="Ratios 3 valores que somam 1.0 (default 0.7 0.15 0.15)")
    ap.add_argument("--preset", type=str, default=None,
                    help="Preset de features: baseline | all | custom (padrão: do YAML ou baseline)")
    ap.add_argument("--features", type=str, default=None,
                    help="Lista CSV de colunas quando --preset custom, ex: abertura,fechamento,volume")
    ap.add_argument("--config", type=str, default=None,
                    help="Caminho opcional para um YAML de configuração")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT,
                    help="Tamanho da janela temporal (seq_len) para o modelo Transformer.")
    args = ap.parse_args()

    par = args.par.upper()
    try:
        tf_s = parse_timeframe(args.timeframe)
    except Exception as e:
        print(f"[ERRO] TIMEFRAME inválido: {e}", flush=True)
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
        preparar(par, tf_s, ratios, preset=preset, features_list=features_list,
                 config_from_yaml=cfg, seq_len=args.seq_len)
        sys.exit(0)
    except Exception as e:
        print(f"[ERRO] Execução interrompida: {e}", flush=True)
        sys.exit(3)
