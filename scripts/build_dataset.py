# -*- coding: utf-8 -*-
"""
build_dataset.py — Transforma OHLCV em features e labels (MVP 1)
Lê CSV bruto de data/raw/{PAR}_M{TFmin}_*.csv e salva em data/processed/.

Uso (CLI):
    python scripts/build_dataset.py EURJPY m5
    python scripts/build_dataset.py EURUSD 300

Parâmetros:
    PAR         -> símbolo (ex.: EURUSD, EURJPY)
    TIMEFRAME   -> m1 | m5 | m15 | 60 | 300 | 900 (ou int)

Saída:
    data/processed/{PAR}_M{TFmin}_features_{nlinhas}x.csv

Regras:
    - Sem vazamento temporal: rótulo usa close(t+1); features usam até t.
    - Cada feature tem função própria.
    - Mantém janela líquida já aplicada no CSV bruto (se vier do extrator).
"""

from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Log sem buffer
# ------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
except Exception:
    pass

# ------------------------------------------------------------
# Caminhos do projeto
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # trans-binary/
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def parse_timeframe(val) -> int:
    """Aceita 'm1', 'm5', 'm15', '60', '300', '900' ou int; retorna segundos."""
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

def tfmin_str(timeframe_s: int) -> str:
    """Converte timeframe em minutos para string M{min} (60s -> M1, 300s -> M5...)."""
    return f"M{timeframe_s // 60}"

def achar_csv_bruto(par: str, timeframe_s: int) -> Optional[Path]:
    """Encontra o CSV bruto mais recente para {par} e {tf} em data/raw."""
    padrao = str(RAW_DIR / f"{par}_{tfmin_str(timeframe_s)}_*.csv")
    candidatos = glob(padrao)
    if not candidatos:
        return None
    # pega o mais novo por mtime
    candidatos.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidatos[0])

def carregar_ohlcv(caminho: Path) -> pd.DataFrame:
    """Lê CSV padrão do extrator e garante tipos."""
    df = pd.read_csv(caminho, parse_dates=["from"])
    # garante ordenação temporal
    df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)
    # garante colunas esperadas
    colunas = ["from", "abertura", "maxima", "minima", "fechamento", "volume"]
    faltando = [c for c in colunas if c not in df.columns]
    if faltando:
        raise ValueError(f"CSV sem colunas esperadas: faltam {faltando}")
    # tipos
    for c in ["abertura", "maxima", "minima", "fechamento", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------------------------------------------
# Features — cada função isola uma família de atributos
# ------------------------------------------------------------
def feat_returns_close_log(df: pd.DataFrame) -> pd.DataFrame:
    """Retornos log do fechamento e variações percentuais simples."""
    out = pd.DataFrame(index=df.index)
    close = df["fechamento"]
    out["ret_log_1"] = np.log(close / close.shift(1))
    out["ret_pct_1"] = (close / close.shift(1)) - 1.0
    out["ret_pct_5"] = (close / close.shift(5)) - 1.0
    out["ret_pct_12"] = (close / close.shift(12)) - 1.0  # ~1h em M5
    return out

def feat_body_wicks(df: pd.DataFrame) -> pd.DataFrame:
    """Tamanho do corpo e pavios; amplitude relativa."""
    out = pd.DataFrame(index=df.index)
    o, h, l, c = df["abertura"], df["maxima"], df["minima"], df["fechamento"]
    body = (c - o)
    range_ = (h - l)
    out["body"] = body
    out["body_abs"] = body.abs()
    out["upper_wick"] = (h - np.maximum(o, c))
    out["lower_wick"] = (np.minimum(o, c) - l)
    out["range"] = range_
    out["range_pct_close"] = range_ / c.shift(1)
    return out

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def feat_ema_cross(df: pd.DataFrame) -> pd.DataFrame:
    """EMAs de curto e médio prazo e cruzamentos."""
    out = pd.DataFrame(index=df.index)
    close = df["fechamento"]
    out["ema_7"] = ema(close, 7)
    out["ema_21"] = ema(close, 21)
    out["ema_50"] = ema(close, 50)
    out["ema_cross_7_21"] = np.sign(out["ema_7"] - out["ema_21"])
    out["ema_cross_21_50"] = np.sign(out["ema_21"] - out["ema_50"])
    return out

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def feat_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[f"rsi_{window}"] = rsi(df["fechamento"], window)
    return out

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def feat_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    tr = true_range(df["maxima"], df["minima"], df["fechamento"])
    out[f"atr_{window}"] = tr.ewm(alpha=1/window, adjust=False).mean()
    return out

def feat_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas móveis do fechamento."""
    out = pd.DataFrame(index=df.index)
    c = df["fechamento"]
    for w in (5, 12, 48):  # ~25min, ~1h, ~4h em M5
        out[f"roll_mean_{w}"] = c.rolling(w).mean()
        out[f"roll_std_{w}"] = c.rolling(w).std()
        out[f"zscore_{w}"] = (c - out[f"roll_mean_{w}"]) / (out[f"roll_std_{w}"] + 1e-12)
    return out

def feat_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Codificação cíclica de hora e dia da semana."""
    out = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["from"], utc=True)
    hour = ts.dt.hour
    dow = ts.dt.dayofweek  # 0=Mon
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)
    return out

# ------------------------------------------------------------
# Labels (sem vazamento): próximo fechamento vs fechamento atual
# ------------------------------------------------------------
def construir_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    next_close = df["fechamento"].shift(-1)
    cur_close = df["fechamento"]
    out["y_up"] = (next_close > cur_close).astype("Int64")
    out["y_down"] = (next_close < cur_close).astype("Int64")
    out["y_noop"] = (next_close == cur_close).astype("Int64")
    return out

# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
def transformar(par: str, timeframe_s: int) -> Tuple[pd.DataFrame, Path]:
    csv = achar_csv_bruto(par, timeframe_s)
    if not csv:
        raise FileNotFoundError(f"Nenhum CSV bruto encontrado para {par} {tfmin_str(timeframe_s)} em {RAW_DIR}")
    print(f"[INFO] Usando bruto: {csv}", flush=True)
    base = carregar_ohlcv(csv)

    # Construção das features
    feats = [
        feat_returns_close_log(base),
        feat_body_wicks(base),
        feat_ema_cross(base),
        feat_rsi(base, 14),
        feat_atr(base, 14),
        feat_rolling_stats(base),
        feat_time_cyclical(base),
    ]
    X = pd.concat(feats, axis=1)

    # Labels
    Y = construir_labels(base)

    # Junta tudo com índices alinhados e remove linhas inválidas (NaN iniciais e a última do label)
    df_all = pd.concat([base[["from", "abertura", "maxima", "minima", "fechamento", "volume"]], X, Y], axis=1)
    # drop primeiros NaN (por janelas) e a última (por y_* shift -1)
    df_all = df_all.dropna().copy()

    # Saída com metadado de linhas
    nlin = len(df_all)
    outpath = PROC_DIR / f"{par}_{tfmin_str(timeframe_s)}_features_{nlin}x.csv"
    df_all.to_csv(outpath, index=False)
    print(f"[OK] Transformado e salvo: {outpath}", flush=True)
    return df_all, outpath

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def _print_uso():
    print("Uso: python scripts/build_dataset.py PAR TIMEFRAME")
    print("Ex.: python scripts/build_dataset.py EURJPY m5")
    print("     python scripts/build_dataset.py EURUSD 300")

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        _print_uso()
        sys.exit(1)

    par = sys.argv[1].upper()
    tf_cli = sys.argv[2]
    try:
        timeframe_s = parse_timeframe(tf_cli)
    except Exception as e:
        print(f"[ERRO] {e}", flush=True)
        _print_uso()
        sys.exit(2)

    try:
        df_out, path_out = transformar(par, timeframe_s)
        # mostra o final para inspeção rápida
        print(df_out.tail(), flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[ERRO] Execução interrompida: {e}", flush=True)
        sys.exit(3)
