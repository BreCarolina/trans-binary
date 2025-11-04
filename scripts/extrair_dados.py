# -*- coding: utf-8 -*-
"""
extrair_dados.py — Extrai candles da IQ Option e salva em data/raw.

Uso (CLI):
    python scripts/extrair_dados.py EURUSD m5 30
    python scripts/extrair_dados.py EURJPY 300 10

Parâmetros:
    PAR         -> símbolo (ex.: EURUSD, EURJPY)
    TIMEFRAME   -> m1 | m5 | m15 | 60 | 300 | 900 (ou int)
    DIAS        -> quantidade de dias a baixar (int)

Credenciais:
    Defina no arquivo .env na raiz do projeto:
        IQ_EMAIL=seu_email
        IQ_PASSWORD=sua_senha

Comportamento:
    - Conecta na IQ Option
    - Baixa candles até ~dias * (86400/timeframe)
    - Descarta candles em formação
    - Filtra janela 07:00–21:00 UTC
    - Salva/atualiza CSV em data/raw: {PAR}_M{TFmin}_{dias_efetivos}d.csv
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from dotenv import load_dotenv

# ------------------------------------------------------------
# Saída não-bufferizada para log em tempo real (CLI/GUI)
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
SCRIPTS_DIR = ROOT / "scripts"                      # trans-binary/scripts/
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Garante que possamos importar pacotes colocados em scripts/
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def parse_timeframe(val) -> int:
    """
    Aceita 'm1', 'm5', 'm15', '60', '300', '900' ou int.
    Retorna sempre segundos (int) para a IQ Option.
    """
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
    raise ValueError(f"TIMEFRAME inválido: {val}. Use m1/m5/m15 ou segundos (60/300/900).")

def _epoch_ultimo_candle_fechado(timeframe: int) -> int:
    """Retorna o epoch alinhado ao último candle completamente fechado."""
    agora = int(time.time())
    return agora - (agora % timeframe)

def _descarta_candles_em_formacao(df: pd.DataFrame, timeframe: int) -> pd.DataFrame:
    """Remove candles cujo fechamento ainda não ocorreu (compara com 'now' UTC)."""
    if df.empty:
        return df
    limite = pd.Timestamp.now(tz="UTC")
    fim_candle = df["from"] + pd.to_timedelta(timeframe, unit="s")
    return df[fim_candle <= limite]

def _retenta(fn, tentativas: int = 3, espera: float = 1.0):
    """Executa uma função com retentativas simples."""
    for i in range(tentativas):
        try:
            return fn()
        except Exception as e:
            if i < tentativas - 1:
                print(f"[AVISO] Falha na chamada (tentativa {i+1}/{tentativas}): {e}", flush=True)
                time.sleep(espera)
            else:
                raise

# ------------------------------------------------------------
# .env e credenciais
# ------------------------------------------------------------
load_dotenv(dotenv_path=ROOT / ".env", override=True)
IQ_EMAIL_ENV = os.getenv("IQ_EMAIL")
IQ_PASSWORD_ENV = os.getenv("IQ_PASSWORD")

# ------------------------------------------------------------
# IQ Option API import shim
# ------------------------------------------------------------
IQAPI_PACKAGE = None
try:
    # Tenta pacote via pip primeiro
    from iqoptionapi.stable_api import IQ_Option as _IQOption
    IQAPI_PACKAGE = "iqoptionapi (pip)"
except Exception:
    # Fallback para pacote local scripts/iqoptionaapi se existir
    try:
        import iqoptionaapi as _iqpkg  # pacote local dentro de scripts/
        sys.modules.setdefault("iqoptionapi", _iqpkg)  # alias de compatibilidade
        from iqoptionaapi.stable_api import IQ_Option as _IQOption
        IQAPI_PACKAGE = "iqoptionaapi (local)"
    except Exception as e:
        raise ImportError(f"Falha ao importar IQ_Option de iqoption(a)api. Detalhe: {e}")

print(f"[INFO] IQ Option API carregada de: {IQAPI_PACKAGE}", flush=True)

# ------------------------------------------------------------
# IQ Option helpers
# ------------------------------------------------------------
def conectar_api(email: Optional[str], senha: Optional[str]) -> "_IQOption":
    """Conecta à IQ Option usando e-mail/senha do .env."""
    email = email or IQ_EMAIL_ENV
    senha = senha or IQ_PASSWORD_ENV
    if not email or not senha:
        raise RuntimeError("Credenciais ausentes. Defina IQ_EMAIL e IQ_PASSWORD no .env.")
    api = _IQOption(email, senha)
    ok, _ = api.connect()
    if not ok:
        raise RuntimeError("Falha na conexão com a IQ Option.")
    print("[INFO] Conectado com sucesso.", flush=True)
    return api

def buscar_candles(api: "_IQOption", par: str, timeframe: int, total: int) -> pd.DataFrame:
    """
    Baixa candles em blocos de até 1000 a partir do último candle fechado,
    caminhando para trás no tempo.
    """
    max_por_chamada = 1000
    todos: List[dict] = []
    cursor = _epoch_ultimo_candle_fechado(timeframe)

    while total > 0:
        qtd = min(max_por_chamada, total)

        def _get():
            return api.get_candles(par, timeframe, qtd, cursor)

        candles = _retenta(_get, tentativas=3, espera=1.0)
        if not candles:
            print("[AVISO] Nenhum bloco retornado nesta iteração. Interrompendo.", flush=True)
            break

        todos.extend(candles)
        try:
            cursor = int(candles[0]["from"]) - 1
        except Exception:
            cursor = candles[0]["from"] - 1
        total -= qtd
        print(f"[INFO] {len(todos)} candles acumulados...", flush=True)

    if not todos:
        return pd.DataFrame()

    df = pd.DataFrame(todos)

    # Normaliza colunas e tipos
    df["from"] = pd.to_datetime(df["from"], unit="s", utc=True)
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df.rename(columns={
        "open": "abertura",
        "max": "maxima",
        "min": "minima",
        "close": "fechamento",
    })

    colunas = ["from", "abertura", "maxima", "minima", "fechamento", "volume"]
    df = df[colunas]
    df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)

    # Exclui candles em formação
    df = _descarta_candles_em_formacao(df, timeframe)
    return df

def filtrar_periodo_liquido(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém somente 07:00–21:00 UTC."""
    if df.empty:
        return df
    return df.set_index("from").between_time("07:00", "21:00").reset_index()

# ------------------------------------------------------------
# Orquestração
# ------------------------------------------------------------
def extrair_dados(
    par: str,
    timeframe: int | str,
    dias: int,
    email: Optional[str] = None,
    senha: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Extrai candles e salva CSV em data/raw."""
    try:
        timeframe = parse_timeframe(timeframe)
    except Exception as e:
        print(f("[ERRO] {e}"), flush=True)
        return pd.DataFrame(), None

    if not par or not timeframe or not dias:
        print("[ERRO] Parâmetros inválidos.", flush=True)
        return pd.DataFrame(), None

    candles_por_dia = (24 * 60 * 60) // int(timeframe)
    total = int(dias) * candles_por_dia
    tf_min = timeframe // 60
    print(f"[INFO] Baixando ~{total} candles: {par} em M{tf_min} por {dias} dia(s).", flush=True)

    api = conectar_api(email, senha)
    df = buscar_candles(api, par, timeframe, total)
    if df.empty:
        print("[AVISO] Nenhum candle retornado após a busca.", flush=True)
        return df, None

    df = filtrar_periodo_liquido(df)
    if df.empty:
        print("[AVISO] Após filtro 07:00–21:00 UTC não restaram candles.", flush=True)
        return df, None

    # Dias efetivos no intervalo resultante (mínimo 1)
    periodo = df["from"].max() - df["from"].min()
    dias_efetivos = max(1, getattr(periodo, "days", 1))

    saida = RAW_DIR / f"{par}_M{tf_min}_{dias_efetivos}d.csv"

    # Se já existe, atualiza incrementalmente
    if saida.exists():
        try:
            antigo = pd.read_csv(saida, parse_dates=["from"])
            antigo = antigo.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)
            df = pd.concat([antigo, df], ignore_index=True)
            df = df.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)
        except Exception as e:
            print(f"[AVISO] Não foi possível mesclar com arquivo existente ({e}). Reescrevendo.", flush=True)

    df.to_csv(saida, index=False)
    print(f"[OK] Salvo: {saida}", flush=True)
    return df, saida

# ------------------------------------------------------------
# Execução direta (CLI)
# ------------------------------------------------------------
def _print_uso():
    print("Uso: python scripts/extrair_dados.py PAR TIMEFRAME DIAS")
    print("Ex.: python scripts/extrair_dados.py EURUSD m5 60")
    print("     python scripts/extrair_dados.py EURJPY 300 10")

if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
        _print_uso()
        sys.exit(1)

    par = sys.argv[1].upper()
    timeframe_cli = sys.argv[2]
    try:
        dias = int(sys.argv[3])
    except Exception:
        print("[ERRO] DIAS deve ser inteiro.", flush=True)
        _print_uso()
        sys.exit(1)

    try:
        dados, caminho = extrair_dados(
            par=par,
            timeframe=timeframe_cli,
            dias=dias,
            email=IQ_EMAIL_ENV,
            senha=IQ_PASSWORD_ENV,
        )
        if not dados.empty:
            print(dados.tail(), flush=True)
            sys.exit(0)
        else:
            sys.exit(2)
    except Exception as e:
        print(f"[ERRO] Execução interrompida: {e}", flush=True)
        sys.exit(3)
