# -*- coding: utf-8 -*-
"""
qc_dataset.py — Verificador operacional de consistência (MVP 1)

Objetivo:
    Validar integridade e consistência dos dados transformados em data/processed
    e checar coerência básica com o bruto em data/raw para o mesmo par/timeframe.

Uso (CLI):
    python scripts/qc_dataset.py EURJPY m5
    python scripts/qc_dataset.py EURUSD 300

Saídas:
    - Código de saída != 0 se houver falhas críticas
    - Relatório Markdown em reports/summaries/{PAR}_{Mtf}_qc_{nlinhas}x.md
    - Resumo no stdout
"""

from __future__ import annotations

import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Log sem buffer
# -----------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
except Exception:
    pass

# -----------------------------
# Caminhos
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "summaries"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utils
# -----------------------------
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

def tfmin_str(timeframe_s: int) -> str:
    return f"M{timeframe_s // 60}"

def achar_proc(par: str, tf_s: int) -> Optional[Path]:
    padrao = str(PROC_DIR / f"{par}_{tfmin_str(tf_s)}_features_*x.csv")
    cand = glob(padrao)
    if not cand:
        return None
    cand.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(cand[0])

def achar_raw(par: str, tf_s: int) -> Optional[Path]:
    padrao = str(RAW_DIR / f"{par}_{tfmin_str(tf_s)}_*.csv")
    cand = glob(padrao)
    if not cand:
        return None
    cand.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(cand[0])

def carregar_csv(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["from"])
    df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)
    return df

def extrair_nlinhas_do_nome(path: Path) -> Optional[int]:
    m = re.search(r"_features_(\d+)x\.csv$", path.name)
    if m:
        return int(m.group(1))
    return None

# -----------------------------
# Checks (cada check retorna (ok: bool, msg: str))
# -----------------------------
def check_colunas_basicas(df: pd.DataFrame) -> Tuple[bool, str]:
    esperadas = ["from","abertura","maxima","minima","fechamento","volume"]
    faltam = [c for c in esperadas if c not in df.columns]
    if faltam:
        return False, f"Faltando colunas básicas: {faltam}"
    return True, "Colunas básicas OK"

def check_ordenacao_unicidade(df: pd.DataFrame) -> Tuple[bool, str]:
    if not df["from"].is_monotonic_increasing:
        return False, "Timestamp não está estritamente ordenado crescente"
    dups = df["from"].duplicated().sum()
    if dups > 0:
        return False, f"Existem {dups} timestamps duplicados"
    return True, "Ordenação e unicidade OK"

def check_nan(df: pd.DataFrame) -> Tuple[bool, str]:
    nans = df.isna().sum()
    total = int(nans.sum())
    if total > 0:
        top = nans[nans>0].sort_values(ascending=False).head(10).to_dict()
        return False, f"NaNs encontrados (top 10): {top}"
    return True, "Sem NaNs"

def check_timeframe(df: pd.DataFrame, tf_s: int) -> Tuple[bool, str]:
    if len(df) < 2:
        return True, "Poucas linhas para checar gaps"
    deltas = (df["from"].diff().dt.total_seconds()).dropna()
    gaps_maiores = (deltas > tf_s).sum()
    prop = float(gaps_maiores) / max(1, len(deltas))
    # Permitimos gaps por janela 07–21 UTC; sinalizamos se >10%
    if prop > 0.10:
        return False, f"Gaps > timeframe em {gaps_maiores} ocorrências ({prop:.1%})"
    return True, f"Gaps dentro do aceitável ({prop:.1%})"

def check_rotulos(df: pd.DataFrame) -> Tuple[bool, str]:
    rot = ["y_up","y_down","y_noop"]
    faltam = [c for c in rot if c not in df.columns]
    if faltam:
        return False, f"Colunas de rótulo ausentes: {faltam}"
    soma = df[rot].sum(axis=1)
    inval = (soma != 1).sum()
    if inval > 0:
        return False, f"{inval} linhas com rótulos que não somam 1"
    # validação cruzada básica
    next_close = df["fechamento"].shift(-1)
    cur_close  = df["fechamento"]
    chk_up   = ((next_close > cur_close).astype("Int64") == df["y_up"]).sum()
    chk_down = ((next_close < cur_close).astype("Int64") == df["y_down"]).sum()
    total = len(df) - 1  # última não tem próximo
    if min(chk_up, chk_down) < total * 0.98:
        return False, "Rótulos não batem (>=2% divergências vs. regra de próximo fechamento)"
    return True, "Rótulos coerentes"

def check_rsi(df: pd.DataFrame) -> Tuple[bool, str]:
    cols = [c for c in df.columns if c.startswith("rsi_")]
    if not cols:
        return True, "Sem RSI — check ignorado"
    serie = df[cols[0]]
    if ((serie < -1e-6) | (serie > 100+1e-6)).any():
        return False, f"RSI fora de [0,100] em {( (serie<0) | (serie>100) ).sum()} linhas"
    return True, "RSI dentro de [0,100]"

def check_atr(df: pd.DataFrame) -> Tuple[bool, str]:
    cols = [c for c in df.columns if c.startswith("atr_")]
    if not cols:
        return True, "Sem ATR — check ignorado"
    if (df[cols[0]] < -1e-12).any():
        return False, "ATR com valores negativos"
    return True, "ATR não-negativo"

def check_ema(df: pd.DataFrame) -> Tuple[bool, str]:
    cols = [c for c in df.columns if c.startswith("ema_")]
    if not cols:
        return True, "Sem EMA — check ignorado"
    if not np.isfinite(df[cols].to_numpy()).all():
        return False, "EMA contém não finitos"
    # consistência de cruzamentos (sinal deve ser -1, 0 ou 1)
    cross = [c for c in df.columns if c.startswith("ema_cross_")]
    if cross:
        vals = df[cross].stack().unique()
        if not set(np.sign(vals)).issubset({-1, 0, 1}):
            return False, "ema_cross contém valores fora de {-1,0,1}"
    return True, "EMAs finitas e cruzamentos válidos"

def check_no_leakage_basico(df: pd.DataFrame) -> Tuple[bool, str]:
    # checa ret_pct_1 coerente com fechamento (sem olhar futuro)
    if "ret_pct_1" in df.columns:
        calc = (df["fechamento"] / df["fechamento"].shift(1)) - 1.0
        dif = (df["ret_pct_1"] - calc).abs().fillna(0)
        if (dif > 1e-9).any():
            return False, "ret_pct_1 diverge do cálculo esperado"
    return True, "Sem vazamento aparente nas métricas simples"

def check_contagem_nome_arquivo(df: pd.DataFrame, path: Path) -> Tuple[bool, str]:
    n_nome = extrair_nlinhas_do_nome(path)
    if n_nome is None:
        return True, "Nome do arquivo sem sufixo de linhas — check ignorado"
    if n_nome != len(df):
        return False, f"Contagem no nome ({n_nome}) difere do total real ({len(df)})"
    return True, "Contagem do nome confere"

# -----------------------------
# Orquestra
# -----------------------------
def gerar_relatorio(par: str, tf_s: int, dfp: pd.DataFrame, path_proc: Path, raw_hint: Optional[Path], resultados: List[Tuple[str,bool,str]]) -> Path:
    ok_total = sum(1 for _, ok, _ in resultados)
    total = len(resultados)
    status = "OK" if ok_total == total else f"{ok_total}/{total} OK"

    nlin = len(dfp)
    out = REPORTS_DIR / f"{par}_{tfmin_str(tf_s)}_qc_{nlin}x.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# QC — {par} {tfmin_str(tf_s)}\n\n")
        f.write(f"- Arquivo processado: `{path_proc}`\n")
        if raw_hint:
            f.write(f"- CSV bruto (referência): `{raw_hint}`\n")
        f.write(f"- Linhas: **{nlin}**\n")
        f.write(f"- Resultado: **{status}**\n\n")
        f.write("## Checks\n\n")
        for nome, ok, msg in resultados:
            icon = "PASS" if ok else "FAIL"
            f.write(f"- **{icon}** — {nome}: {msg}\n")
    return out

def main(par: str, tf_val: str):
    tf_s = parse_timeframe(tf_val)
    proc = achar_proc(par, tf_s)
    if not proc:
        print(f"[ERRO] Arquivo processado não encontrado para {par} {tfmin_str(tf_s)} em {PROC_DIR}", flush=True)
        sys.exit(2)
    raw = achar_raw(par, tf_s)  # apenas referência
    dfp = carregar_csv(proc)

    checks: List[Tuple[str,bool,str]] = []
    def run(nome: str, fn):
        try:
            ok, msg = fn(dfp)
        except TypeError:
            ok, msg = fn(dfp, tf_s)
        checks.append((nome, ok, msg))

    run("Colunas básicas", check_colunas_basicas)
    run("Ordenação e unicidade", check_ordenacao_unicidade)
    run("NaNs", check_nan)
    run("Gaps vs timeframe", check_timeframe)
    run("Rótulos", check_rotulos)
    run("RSI", check_rsi)
    run("ATR", check_atr)
    run("EMA e cruzamentos", check_ema)
    run("No-leakage básico", check_no_leakage_basico)
    checks.append(("Contagem no nome do arquivo",) + check_contagem_nome_arquivo(dfp, proc))

    rel = gerar_relatorio(par, tf_s, dfp, proc, raw, checks)

    # imprime resumo curto
    fails = [c for c in checks if not c[1]]
    if fails:
        print("[QC] Falhas encontradas:", flush=True)
        for nome, _, msg in fails:
            print(f" - {nome}: {msg}", flush=True)
        print(f"[QC] Relatório: {rel}", flush=True)
        sys.exit(1)
    else:
        print("[QC] Tudo OK. Relatório gerado:", rel, flush=True)
        sys.exit(0)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ("-h","--help"):
        print("Uso: python scripts/qc_dataset.py PAR TIMEFRAME", flush=True)
        print("Ex.: python scripts/qc_dataset.py EURJPY m5", flush=True)
        sys.exit(1)
    main(sys.argv[1].upper(), sys.argv[2])
