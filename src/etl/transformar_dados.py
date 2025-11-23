# -*- coding: utf-8 -*-
"""
transformar_dados.py — Transforma OHLCV em features e rótulos.

Comportamento:
- Lê o CSV bruto mais recente para o par/timeframe em data/raw/
- Gera um conjunto completo e consistente de features (tempo, retornos, corpo/pavios,
  EMAs, RSI, ATR, rolling stats)
- Constrói rótulos sem vazamento (horizonte = 1 candle)
- Remove linhas com NaNs geradas por janelas / indicadores
- Salva o dataset pronto em data/processed/ com nome informativo

CLI (exemplo):
    python3 src/etl/transformar_dados.py EURJPY m5

Este script gera o material completo.
"""

from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Saída sem buffer
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
except Exception:
    pass

# Caminhos do projeto (raiz onde está .env e pasta src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Torna src importável quando executado como 'python3 src/etl/transformar_dados.py'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Constants
LABEL_HORIZON_CANDLES = 1
LABEL_THRESHOLD_RET = 0.0005

# -------------------- Utils --------------------

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
    return f"M{int(timeframe_s) // 60}"


def achar_csv_bruto(par: str, timeframe_s: int) -> Optional[Path]:
    padrao = str(RAW_DIR / f"{par}_{tfmin_str(timeframe_s)}_*.csv")
    candidatos = glob(padrao)
    if not candidatos:
        return None
    candidatos.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidatos[0])


def carregar_ohlcv(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["from"])
    df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)
    colunas = ["from", "abertura", "maxima", "minima", "fechamento", "volume"]
    faltando = [c for c in colunas if c not in df.columns]
    if faltando:
        raise ValueError(f"CSV sem colunas esperadas: faltam {faltando}")
    for c in ["abertura", "maxima", "minima", "fechamento", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[colunas]

# -------------------- Feature builders (completos) --------------------

def feat_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["from"], utc=True)
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    out = pd.DataFrame(index=df.index)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def feat_returns(df: pd.DataFrame) -> pd.DataFrame:
    c = df["fechamento"]
    out = pd.DataFrame(index=df.index)
    out["ret_pct_1"] = c.pct_change(1)
    out["ret_pct_5"] = c.pct_change(5)
    out["ret_log_1"] = np.log(c / c.shift(1))
    return out


def feat_body_wicks(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["abertura"], df["maxima"], df["minima"], df["fechamento"]
    out = pd.DataFrame(index=df.index)
    body = (c - o)
    out["body"] = body
    out["body_abs"] = body.abs()
    out["upper_wick"] = (h - np.maximum(o, c))
    out["lower_wick"] = (np.minimum(o, c) - l)
    out["range"] = (h - l)
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def feat_emas_rsi_atr(df: pd.DataFrame) -> pd.DataFrame:
    c = df["fechamento"]
    out = pd.DataFrame(index=df.index)
    out["ema_7"] = ema(c, 7)
    out["ema_21"] = ema(c, 21)
    # RSI
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder-like smoothing using ewm with alpha=1/14
    rup = up.ewm(alpha=1/14, adjust=False).mean()
    rdn = down.ewm(alpha=1/14, adjust=False).mean()
    rs = rup / (rdn.replace(0, np.nan))
    out["rsi_14"] = 100 - (100 / (1 + rs))
    # ATR
    prev_c = c.shift(1)
    tr = pd.concat([(df["maxima"] - df["minima"]),
                    (df["maxima"] - prev_c).abs(),
                    (df["minima"] - prev_c).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return out


def feat_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    c = df["fechamento"]
    out = pd.DataFrame(index=df.index)
    for w in (5, 12, 48):
        out[f"roll_mean_{w}"] = c.rolling(w).mean()
        out[f"roll_std_{w}"] = c.rolling(w).std()
    return out

# -------------------- Labels (sem vazamento) --------------------

def construir_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    next_close = df["fechamento"].shift(-LABEL_HORIZON_CANDLES)
    cur_close = df["fechamento"]
    diff = next_close - cur_close

    direction_bin = (diff > 0).astype("Int64")
    direction_ternary = np.sign(diff).astype("Int64")
    out["direction_bin"] = direction_bin
    out["direction_ternary"] = direction_ternary
    out["target_bin"] = direction_bin

    threshold = LABEL_THRESHOLD_RET
    ret = diff / cur_close
    y_up = (ret > threshold).astype("Int64")
    y_down = (ret < -threshold).astype("Int64")
    y_noop = ((ret >= -threshold) & (ret <= threshold)).astype("Int64")
    out["y_up"] = y_up
    out["y_down"] = y_down
    out["y_noop"] = y_noop
    out["target_ternary"] = (y_up - y_down).astype("Int64")
    return out

# -------------------- Pipeline --------------------

def transformar(par: str, timeframe_s: int) -> Tuple[pd.DataFrame, Path]:
    csv = achar_csv_bruto(par, timeframe_s)
    if not csv:
        raise FileNotFoundError(f"Nenhum CSV bruto encontrado para {par} {tfmin_str(timeframe_s)} em {RAW_DIR}")
    print(f"[INFO] Usando bruto: {csv}", flush=True)

    base = carregar_ohlcv(csv)

    # Gera todas as features (completas)
    feats: List[pd.DataFrame] = []
    for fn in (feat_time_cyclical, feat_returns, feat_body_wicks, feat_rolling_stats, feat_emas_rsi_atr):
        fx = fn(base)
        if not isinstance(fx, pd.DataFrame):
            raise TypeError(f"Função de feature {fn.__name__} não retornou DataFrame.")
        feats.append(fx)

    X = pd.concat(feats, axis=1) if feats else pd.DataFrame(index=base.index)

    Y = construir_labels(base)

    cols_base = ["from", "abertura", "maxima", "minima", "fechamento", "volume"]
    df_all = pd.concat([base[cols_base], X, Y], axis=1)

    # Remover NaNs originados por indicadores/janelas e remover as últimas linhas
    df_all = df_all.dropna().iloc[: -LABEL_HORIZON_CANDLES if LABEL_HORIZON_CANDLES > 0 else None].copy()

    nlin = len(df_all)
    outpath = PROC_DIR / f"{par}_{tfmin_str(timeframe_s)}_full_{nlin}x.csv"
    df_all.to_csv(outpath, index=False)
    print(f"[OK] Transformado e salvo: {outpath}", flush=True)
    print(f"[INFO] Linhas: {nlin}  Colunas: {len(df_all.columns)}", flush=True)
    return df_all, outpath

# -------------------- CLI --------------------

def _print_uso():
    print("Uso: python3 src/etl/transformar_dados.py PAR TIMEFRAME", flush=True)
    print("Ex.: python3 src/etl/transformar_dados.py EURJPY m5", flush=True)


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        _print_uso()
        sys.exit(1)

    par = sys.argv[1].upper()
    timeframe_cli = sys.argv[2]
    try:
        timeframe_s = parse_timeframe(timeframe_cli)
    except Exception as e:
        print(f"[ERRO] {e}", flush=True)
        sys.exit(2)

    try:
        df_out, path_out = transformar(par, timeframe_s)
        print(df_out.tail(), flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[ERRO] Execução interrompida: {e}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
