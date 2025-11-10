# -*- coding: utf-8 -*-
"""
transformar_dados.py — Transforma OHLCV em features e rótulos (MVP1, revisado)

Planejamento:
- Baseline = somente colunas extraídas (OHLCV) + codificação de tempo (seno/cosseno).
- Demais famílias de features ficam disponíveis, mas só entram se a preset escolhida pedir.
- Sem vazamento temporal: rótulo usa close(t+1); features usam até t.

Uso (CLI):
    python scripts/transformar_dados.py EURJPY m5
    python scripts/transformar_dados.py EURUSD 300 --preset tech-lite
    python scripts/transformar_dados.py EURUSD m5 --list-presets

Saída:
    data/processed/{PAR}_{Mxx}_{preset}_{nlinhas}x.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Saída sem buffer (útil para logs em tempo real)
# ---------------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
except Exception:
    pass

# ---------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # trans-binary/
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
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
    """Converte timeframe em minutos para 'M{min}' (60s -> M1, 300s -> M5...)."""
    return f"M{int(timeframe_s) // 60}"

def achar_csv_bruto(par: str, timeframe_s: int) -> Optional[Path]:
    """Encontra o CSV bruto mais recente para {par} e {tf} em data/raw."""
    padrao = str(RAW_DIR / f"{par}_{tfmin_str(timeframe_s)}_*.csv")
    candidatos = glob(padrao)
    if not candidatos:
        return None
    candidatos.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidatos[0])

def carregar_ohlcv(caminho: Path) -> pd.DataFrame:
    """Lê CSV padrão do extrator e garante tipos/ordem."""
    df = pd.read_csv(caminho, parse_dates=["from"])
    df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)
    colunas = ["from", "abertura", "maxima", "minima", "fechamento", "volume"]
    faltando = [c for c in colunas if c not in df.columns]
    if faltando:
        raise ValueError(f"CSV sem colunas esperadas: faltam {faltando}")
    for c in ["abertura", "maxima", "minima", "fechamento", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[colunas]

# ---------------------------------------------------------------------
# Família de features (opcionais; baseline usa só OHLCV + tempo)
# ---------------------------------------------------------------------
def feat_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Codificação cíclica de hora e dia da semana (sem vazamento)."""
    out = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["from"], utc=True)
    hour = ts.dt.hour
    dow = ts.dt.dayofweek  # 0=Mon
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)
    return out

def feat_returns_close_log(df: pd.DataFrame) -> pd.DataFrame:
    """Retornos do fechamento (excluído do baseline)."""
    out = pd.DataFrame(index=df.index)
    c = df["fechamento"]
    out["ret_log_1"] = np.log(c / c.shift(1))
    out["ret_pct_1"] = (c / c.shift(1)) - 1.0
    out["ret_pct_5"] = (c / c.shift(5)) - 1.0
    return out

def feat_body_wicks(df: pd.DataFrame) -> pd.DataFrame:
    """Corpo e pavios (excluído do baseline)."""
    out = pd.DataFrame(index=df.index)
    o, h, l, c = df["abertura"], df["maxima"], df["minima"], df["fechamento"]
    body = (c - o)
    out["body"] = body
    out["body_abs"] = body.abs()
    out["upper_wick"] = (h - np.maximum(o, c))
    out["lower_wick"] = (np.minimum(o, c) - l)
    out["range"] = (h - l)
    return out

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def feat_ema_rsi_atr(df: pd.DataFrame) -> pd.DataFrame:
    """EMAs, RSI e ATR (excluídos do baseline)."""
    out = pd.DataFrame(index=df.index)
    c = df["fechamento"]
    # EMAs
    out["ema_7"] = ema(c, 7)
    out["ema_21"] = ema(c, 21)
    # RSI
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
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
    """Estatísticas móveis (excluídas do baseline)."""
    out = pd.DataFrame(index=df.index)
    c = df["fechamento"]
    for w in (5, 12, 48):
        out[f"roll_mean_{w}"] = c.rolling(w).mean()
        out[f"roll_std_{w}"] = c.rolling(w).std()
    return out

# ---------------------------------------------------------------------
# Presets de features
# ---------------------------------------------------------------------
# Cada preset é uma lista de funções que recebem df(base OHLCV) e retornam DataFrame de features.
FEATURE_PRESETS: Dict[str, List[Callable[[pd.DataFrame], pd.DataFrame]]] = {
    # Baseline: somente OHLCV + tempo (nenhuma derivada de preço)
    "baseline": [
        feat_time_cyclical,
    ],
    # Tech-lite: um conjunto leve de técnicas
    "tech-lite": [
        feat_time_cyclical,
        feat_returns_close_log,
        feat_body_wicks,
        feat_rolling_stats,
    ],
    # Tech-full: conjunto mais completo
    "tech-full": [
        feat_time_cyclical,
        feat_returns_close_log,
        feat_body_wicks,
        feat_rolling_stats,
        feat_ema_rsi_atr,
    ],
}

def listar_presets() -> str:
    linhas = []
    for k, fns in FEATURE_PRESETS.items():
        linhas.append(f"- {k}: {', '.join(fn.__name__ for fn in fns)}")
    return "\n".join(linhas)

# ---------------------------------------------------------------------
# Labels (sem vazamento): comparam fechamento(t+1) vs fechamento(t)
# Fornecemos três variantes: binária, ternária e one-hot ternária.
# ---------------------------------------------------------------------
def construir_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    next_close = df["fechamento"].shift(-1)
    cur_close = df["fechamento"]
    diff = next_close - cur_close

    # Binária: up/down
    out["target_bin"] = (diff > 0).astype("Int64")  # 1=up, 0=down/flat

    # Ternária: {-1,0,1}
    out["target_ternary"] = np.sign(diff).astype("Int64")

    # One-hot (3 classes)
    out["y_up"]   = (diff > 0).astype("Int64")
    out["y_down"] = (diff < 0).astype("Int64")
    out["y_noop"] = (diff == 0).astype("Int64")

    return out

# ---------------------------------------------------------------------
# Pipeline de transformação
# ---------------------------------------------------------------------
def transformar(par: str, timeframe_s: int, preset: str = "baseline") -> Tuple[pd.DataFrame, Path]:
    csv = achar_csv_bruto(par, timeframe_s)
    if not csv:
        raise FileNotFoundError(f"Nenhum CSV bruto encontrado para {par} {tfmin_str(timeframe_s)} em {RAW_DIR}")
    print(f"[INFO] Usando bruto: {csv}", flush=True)

    base = carregar_ohlcv(csv)  # mantém ["from","abertura","maxima","minima","fechamento","volume"]

    # Seleção de features por preset
    preset = preset.strip().lower()
    if preset not in FEATURE_PRESETS:
        raise ValueError(f"Preset '{preset}' inválido. Use --list-presets para ver opções.")

    feat_frames: List[pd.DataFrame] = []
    for fn in FEATURE_PRESETS[preset]:
        fx = fn(base)
        if not isinstance(fx, pd.DataFrame):
            raise TypeError(f"Função de feature {fn.__name__} não retornou DataFrame.")
        feat_frames.append(fx)

    X = pd.concat(feat_frames, axis=1) if feat_frames else pd.DataFrame(index=base.index)

    # Rótulos (sem vazamento)
    Y = construir_labels(base)

    # Monta dataset final:
    # - Sempre inclui OHLCV originais (baseline explícito) + X (features do preset) + Y
    # - Remove NaNs e a última linha (sem rótulo por causa do shift -1)
    cols_base = ["from", "abertura", "maxima", "minima", "fechamento", "volume"]
    df_all = pd.concat([base[cols_base], X, Y], axis=1)
    df_all = df_all.dropna().iloc[:-1].copy()

    # Persistência
    nlin = len(df_all)
    outpath = PROC_DIR / f"{par}_{tfmin_str(timeframe_s)}_{preset}_{nlin}x.csv"
    df_all.to_csv(outpath, index=False)
    print(f"[OK] Transformado e salvo: {outpath}", flush=True)

    # Log rápido de colunas
    print(f"[INFO] Colunas ({len(df_all.columns)}): {list(df_all.columns)}", flush=True)
    return df_all, outpath

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transforma OHLCV em dataset com features/labels.")
    p.add_argument("par", type=str, help="Símbolo, ex.: EURUSD, EURJPY")
    p.add_argument("timeframe", type=str, help="m1 | m5 | m15 | 60 | 300 | 900 (ou int)")
    p.add_argument("--preset", type=str, default="baseline",
                   help=f"Preset de features (default: baseline). Use --list-presets para ver opções.")
    p.add_argument("--list-presets", action="store_true", help="Lista presets disponíveis e sai.")
    return p

def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.list_presets:
        print("Presets disponíveis:")
        print(listar_presets())
        sys.exit(0)

    par = args.par.upper()
    try:
        timeframe_s = parse_timeframe(args.timeframe)
    except Exception as e:
        print(f"[ERRO] {e}", flush=True)
        sys.exit(2)

    try:
        df_out, path_out = transformar(par, timeframe_s, preset=args.preset)
        # Preview de cauda para conferência rápida
        print(df_out.tail(), flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[ERRO] Execução interrompida: {e}", flush=True)
        sys.exit(3)

if __name__ == "__main__":
    main()
