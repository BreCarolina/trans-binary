# src/tests/teste_pipeline_dados.py
# -*- coding: utf-8 -*-
"""
Testes básicos de consistência para os artefatos do pipeline:
 - extração -> raw CSV
 - transformação -> processed full CSV
 - preparação -> preset CSVs, .npy e manifest

Rode:
    pytest -q
ou
    python -m pytest -q
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# ----- CONFIG -----
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"
MANI_DIR = REPO_ROOT / "data" / "manifests"

# Defaults that match o que rodamos nos exemplos
PAR = "EURJPY"
TFMIN = "M5"  # formato usado nos arquivos processados
PRESET = "baseline"

# Helper to find latest file by pattern
def latest_by_pattern(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

# --------------------
def test_raw_csv_exists():
    # Procura pelo CSV bruto do par (qualquer quantidade de dias)
    pat = f"{PAR}_M5_*.csv"
    raw = latest_by_pattern(RAW_DIR, pat)
    assert raw is not None, f"Arquivo raw não encontrado com padrão {pat} em {RAW_DIR}"
    df = pd.read_csv(raw, parse_dates=["from"])
    # Checagens mínimas
    for col in ("from", "abertura", "maxima", "minima", "fechamento"):
        assert col in df.columns, f"Coluna esperada '{col}' ausente no raw CSV {raw}"
    assert len(df) > 10, "Raw CSV aparentemente muito pequeno."

def test_transformed_full_exists_and_basic_checks():
    # Procura pelo CSV full gerado pelo transformador
    pat = f"{PAR}_{TFMIN.lower()}_*full_*x.csv".upper()  # tentar pegar variações, mas veremos fallback
    # Simples fallback: buscar qualquer arquivo contendo PAR and 'full'
    candidates = [p for p in (PROC_DIR).glob(f"{PAR}_*") if "full" in p.name.lower()]
    assert candidates, f"Nenhum CSV 'full' encontrado em {PROC_DIR} para {PAR}"
    full_csv = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    df = pd.read_csv(full_csv, parse_dates=["from"])
    # Colunas esperadas
    required = {"from", "abertura", "maxima", "minima", "fechamento", "volume"}
    assert required.issubset(set(df.columns)), f"Colunas base faltando em {full_csv}"
    # Rótulos básicos
    assert any(c in df.columns for c in ("target_bin", "direction_bin", "y_up")), "Rótulos esperados ausentes no CSV full"
    # Sem duplicados na coluna 'from'
    assert df["from"].duplicated().sum() == 0, "'from' contém duplicados"
    # Alguns valores numéricos não-NaN
    numeric_cols = ["abertura", "fechamento"]
    for c in numeric_cols:
        assert pd.to_numeric(df[c], errors="coerce").notna().sum() > 0, f"Coluna {c} sem valores numéricos válidos"

def test_prepared_artifacts_exist_and_manifest():
    preset_dir = PROC_DIR / PRESET
    assert preset_dir.exists() and preset_dir.is_dir(), f"Diretório do preset não encontrado: {preset_dir}"

    prefix = f"{PAR}_{TFMIN.lower()}".upper()
    # arquivos CSV esperados
    X_train_csv = latest_by_pattern(preset_dir, f"{PAR}_*_{PRESET}_X_train.csv")
    y_train_csv = latest_by_pattern(preset_dir, f"{PAR}_*_{PRESET}_y_train.csv")
    assert X_train_csv is not None and y_train_csv is not None, "Arquivos X_train/y_train CSV não encontrados no preset"

    # Manifest
    manifest_pat = f"{PAR}_*_{PRESET}_manifest.json"
    manifest_path = latest_by_pattern(MANI_DIR, manifest_pat)
    assert manifest_path is not None, f"Manifest não encontrado em {MANI_DIR} com padrão {manifest_pat}"

    # Carrega manifesto e checa chaves
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for k in ("features_used", "seq_len", "csv_artifacts", "npy_artifacts", "scaler"):
        assert k in manifest, f"Chave '{k}' ausente no manifest {manifest_path}"

    # CSVs normalizados
    X_train_df = pd.read_csv(X_train_csv)
    y_train_df = pd.read_csv(y_train_csv)
    assert len(X_train_df) == len(y_train_df), "X_train e y_train têm números diferentes de linhas"
    assert X_train_df.isna().sum().sum() == 0, "X_train CSV contém NaNs"

def test_npy_sequences_and_shapes():
    # Usa manifesto para localizar npy e seq_len
    manifest_pat = f"{PAR}_*_{PRESET}_manifest.json"
    manifest_path = latest_by_pattern(MANI_DIR, manifest_pat)
    assert manifest_path is not None, "Manifest não encontrado para checar NPZ/NPY"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seq_len = int(manifest.get("seq_len", 60))
    npy_artifacts = manifest.get("npy_artifacts") or {}
    assert npy_artifacts, "Sem npy_artifacts no manifest"

    # Carrega X_train.npy e y_train.npy
    x_train_npy = Path(npy_artifacts.get("X_train_npy"))
    y_train_npy = Path(npy_artifacts.get("y_train_npy"))
    assert x_train_npy.exists() and y_train_npy.exists(), f"Arquivos .npy esperados não encontrados: {x_train_npy}, {y_train_npy}"

    X_train_arr = np.load(str(x_train_npy))
    y_train_arr = np.load(str(y_train_npy))
    assert X_train_arr.ndim == 3, "X_train.npy deveria ter 3 dimensões (N, seq_len, n_features)"
    assert X_train_arr.shape[1] == seq_len, f"Dimensão seq_len ({X_train_arr.shape[1]}) não bate com seq_len do manifest ({seq_len})"
    assert X_train_arr.shape[0] == y_train_arr.shape[0], "Primeira dimensão de X_train.npy deve igualar comprimento de y_train.npy"
    # Sem NaNs
    assert not np.isnan(X_train_arr).any(), "X_train.npy contém NaNs"
    assert not np.isnan(y_train_arr).any(), "y_train.npy contém NaNs"

def test_features_consistency_with_manifest():
    # Load manifest
    manifest_pat = f"{PAR}_*_{PRESET}_manifest.json"
    manifest_path = latest_by_pattern(MANI_DIR, manifest_pat)
    assert manifest_path is not None, "Manifest não encontrado"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features_used = manifest.get("features_used", [])
    assert isinstance(features_used, list) and features_used, "features_used inválido no manifest"

    # Checar CSV X_train cabeçalhos
    preset_dir = PROC_DIR / PRESET
    X_train_csv = latest_by_pattern(preset_dir, f"{PAR}_*_{PRESET}_X_train.csv")
    assert X_train_csv is not None, "X_train CSV não encontrado para checagem de colunas"
    Xdf = pd.read_csv(X_train_csv)
    # features_used deve estar contido nas colunas do Xdf (ordem pode variar)
    missing = [f for f in features_used if f not in Xdf.columns]
    assert not missing, f"As features listadas no manifest estão ausentes no CSV X_train: {missing}"

def test_label_values_reasonable():
    # Pega y_train CSV e confere valores
    preset_dir = PROC_DIR / PRESET
    y_train_csv = latest_by_pattern(preset_dir, f"{PAR}_*_{PRESET}_y_train.csv")
    assert y_train_csv is not None, "y_train CSV não encontrado para validação de rótulo"
    ydf = pd.read_csv(y_train_csv)
    # Valores esperados: 0/1 (target binário) ou -1/0/1 para ternário
    uniques = set(ydf.iloc[:, 0].dropna().unique().tolist())
    allowed = {0, 1, -1}
    assert uniques.issubset(allowed), f"Valores incomuns em y_train: {uniques}"

# If running as a script (not via pytest), allow direct execute
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
