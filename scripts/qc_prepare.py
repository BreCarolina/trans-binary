# -*- coding: utf-8 -*-
"""
qc_prepare.py — Verificador de consistência dos splits e normalização (MVP 2)

Valida:
  - Presença e leitura de X_train/X_val/X_test e y_*
  - Colunas idênticas entre X_* e compatíveis com manifesto
  - Conjuntos com NaNs/Inf? (não pode)
  - y_* mutuamente exclusivos (soma == 1)
  - Tamanhos compatíveis com os índices de split do manifesto
  - Normalização: média ~0 e desvio ~1 no X_train (tolerância), e finitude em todos
  - Relatório resumido no stdout e código de saída !=0 em caso de falhas

Uso:
  python scripts/qc_prepare.py EURJPY m5
"""

from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MANI_DIR = ROOT / "data" / "manifests"

def tfmin_str(tf_s: int) -> str:
    return f"M{tf_s // 60}"

def parse_timeframe(val) -> int:
    if isinstance(val, int):
        return val
    s = str(val).strip().lower()
    mapa = {"m1":60,"1m":60,"60":60, "m5":300,"5m":300,"300":300, "m15":900,"15m":900,"900":900}
    if s in mapa: return mapa[s]
    if s.isdigit(): return int(s)
    raise ValueError(f"TIMEFRAME inválido: {val}")

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def finite_ok(df: pd.DataFrame) -> Tuple[bool, str]:
    arr = df.to_numpy()
    if np.isnan(arr).any():
        return False, "Contém NaN"
    if np.isinf(arr).any():
        return False, "Contém Inf/-Inf"
    return True, "Sem NaN/Inf"

def labels_ok(y: pd.DataFrame) -> Tuple[bool, str]:
    lbls = [c for c in y.columns if c.startswith("y_")]
    if not lbls:
        return False, "Colunas y_* ausentes"
    s = y[lbls].sum(axis=1)
    bad = (s != 1).sum()
    if bad > 0:
        return False, f"{bad} linhas com soma de rótulos != 1"
    return True, "Rótulos mutuamente exclusivos"

def columns_match(Xs: List[pd.DataFrame]) -> Tuple[bool, str]:
    cols0 = list(Xs[0].columns)
    for i, Xi in enumerate(Xs[1:], start=1):
        if list(Xi.columns) != cols0:
            return False, f"Colunas diferentes entre X[0] e X[{i}]"
    return True, "Colunas idênticas em X_train/X_val/X_test"

def columns_vs_manifest_ok(cols: List[str], manifest_cols: List[str]) -> Tuple[bool, str]:
    if cols != manifest_cols:
        return False, "Colunas de features divergem do manifesto"
    return True, "Colunas compatíveis com manifesto"

def shape_vs_manifest(n_train: int, n_val: int, n_test: int, mani: dict) -> Tuple[bool, str]:
    idx = mani["split"]["index_ranges"]
    exp_train = idx["train"][1] - idx["train"][0]
    exp_val   = idx["val"][1]   - idx["val"][0]
    exp_test  = idx["test"][1]  - idx["test"][0]
    if (n_train, n_val, n_test) != (exp_train, exp_val, exp_test):
        return False, f"Tamanhos divergentes: obtidos {(n_train,n_val,n_test)} vs manifesto {(exp_train,exp_val,exp_test)}"
    return True, "Tamanhos batem com o manifesto"

def normalization_ok(X_train: pd.DataFrame, tol_mean=0.05, tol_std=0.05) -> Tuple[bool, str]:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    # aceita alguns outliers, mas verifica proporções
    bad_mean = (mu.abs() > tol_mean).sum()
    bad_std  = ((sd - 1).abs() > tol_std).sum()
    if bad_mean > 0 or bad_std > 0:
        return False, f"Normalização fora da faixa: {bad_mean} col(s) |μ|>{tol_mean}, {bad_std} col(s) |σ-1|>{tol_std}"
    return True, "Normalização do TRAIN dentro da faixa esperada"

def main(par: str, tf_val: str):
    tf_s = parse_timeframe(tf_val)
    prefix = f"{par}_{tfmin_str(tf_s)}"
    paths = {
        "X_train": PROC_DIR / f"{prefix}_X_train.csv",
        "y_train": PROC_DIR / f"{prefix}_y_train.csv",
        "X_val":   PROC_DIR / f"{prefix}_X_val.csv",
        "y_val":   PROC_DIR / f"{prefix}_y_val.csv",
        "X_test":  PROC_DIR / f"{prefix}_X_test.csv",
        "y_test":  PROC_DIR / f"{prefix}_y_test.csv",
        "manifest": MANI_DIR / f"{prefix}_manifest.json",
    }
    missing = [k for k,p in paths.items() if not p.exists()]
    if missing:
        print(f"[ERRO] Arquivos ausentes: {missing}", flush=True)
        sys.exit(2)

    with open(paths["manifest"], "r", encoding="utf-8") as f:
        mani = json.load(f)

    X_train = load_csv(paths["X_train"])
    y_train = load_csv(paths["y_train"])
    X_val   = load_csv(paths["X_val"])
    y_val   = load_csv(paths["y_val"])
    X_test  = load_csv(paths["X_test"])
    y_test  = load_csv(paths["y_test"])

    checks = []

    # 1) Colunas e finitude
    ok, msg = columns_match([X_train, X_val, X_test]); checks.append(("Colunas X_*", ok, msg))
    ok, msg = finite_ok(X_train); checks.append(("Finitude X_train", ok, msg))
    ok, msg = finite_ok(X_val);   checks.append(("Finitude X_val", ok, msg))
    ok, msg = finite_ok(X_test);  checks.append(("Finitude X_test", ok, msg))

    # 2) Labels
    ok, msg = labels_ok(y_train); checks.append(("Labels y_train", ok, msg))
    ok, msg = labels_ok(y_val);   checks.append(("Labels y_val", ok, msg))
    ok, msg = labels_ok(y_test);  checks.append(("Labels y_test", ok, msg))

    # 3) Tamanhos vs manifesto
    ok, msg = shape_vs_manifest(len(X_train), len(X_val), len(X_test), mani); checks.append(("Tamanhos vs manifesto", ok, msg))

    # 4) Colunas vs manifesto
    mani_feats = mani["columns"]["features"]
    ok, msg = columns_vs_manifest_ok(list(X_train.columns), mani_feats); checks.append(("Features vs manifesto", ok, msg))

    # 5) Normalização (média≈0, std≈1 no TRAIN)
    ok, msg = normalization_ok(X_train); checks.append(("Normalização TRAIN", ok, msg))

    fails = [(n,m) for n,ok,m in checks if not ok]
    if fails:
        print("[QC-PREP] Falhas encontradas:", flush=True)
        for nome, msg in fails:
            print(f" - {nome}: {msg}", flush=True)
        sys.exit(1)
    else:
        print("[QC-PREP] Tudo OK: splits e normalização consistentes.", flush=True)
        print(f" - X_train: {paths['X_train']}", flush=True)
        print(f" - X_val:   {paths['X_val']}", flush=True)
        print(f" - X_test:  {paths['X_test']}", flush=True)
        print(f" - Manifesto: {paths['manifest']}", flush=True)
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ("-h","--help"):
        print("Uso: python scripts/qc_prepare.py PAR TIMEFRAME", flush=True)
        sys.exit(1)
    main(sys.argv[1].upper(), sys.argv[2])
