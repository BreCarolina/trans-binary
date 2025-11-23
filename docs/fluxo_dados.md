
# Processo de Dados

Documentação objetiva dos três scripts principais do pipeline de dados: **extração**, **transformação** e **preparação**.

## 1. `extrair_dados.py`

### **Caminho**

```
src/etl/extrair_dados.py
```

### **Função / Objetivo**

Conectar à IQ Option, baixar candles brutos para um par e timeframe definidos, filtrar período líquido e salvar um CSV padronizado em `data/raw/`.

### **Funcionamento**

* Carrega credenciais do `.env` na raiz do projeto.
* Conecta à API da IQ Option.
* Baixa candles em blocos (até 1000 por requisição).
* Remove candles em formação.
* Filtra horário 07:00–21:00 UTC.
* Salva o arquivo bruto com nome padronizado.

### **Artefato Gerado**

Local: `data/raw/`

* Exemplo: `EURJPY_M5_298d.csv`


## 2. `transformar_dados.py`

### **Caminho**

```
src/etl/transformar_dados.py
```

### **Função / Objetivo**

Ler o CSV bruto, gerar **todas** as features e rótulos necessários para aprendizado de máquina sem vazamento, resultando em um dataset completo.

### **Funcionamento**

* Carrega o arquivo bruto mais recente.
* Garante ordenação temporal.
* Gera features:

  * Codificação cíclica de tempo.
  * Retornos percentuais e logarítmicos.
  * Corpo e pavios.
  * Rolling stats.
  * EMAs, RSI, ATR.
* Gera rótulos:

  * `direction_bin`, `direction_ternary`, `y_up`, `y_down`, `y_noop`, `target_bin`, `target_ternary`.
* Remove NaNs de indicadores e a última linha sem rótulo.
* Salva um CSV completo.

### **Artefato Gerado**

Local: `data/processed/`

* Exemplo: `EURJPY_M5_full_36286x.csv`


## 3. `preparar_dados.py`

### **Caminho**

```
src/etl/preparar_dados.py
```

### **Função / Objetivo**

Gerar o dataset final utilizado pelo modelo: seleção de features via preset, normalização, splits temporais e criação de janelas (`seq_len`).

### **Funcionamento**

* Carrega o arquivo transformado mais recente.
* Define quais features usar via preset:

  * `baseline` → OHLCV + tempo.
  * `all` → todas as features.
  * `custom` → lista fornecida.
* Realiza splits temporais (train/val/test).
* Normaliza com StandardScaler (fit no treino).
* Cria janelas temporais para o modelo.
* Salva versões CSV e `.npy` dos dados.
* Gera manifesto JSON completo.

### **Artefatos Gerados**

Local: `data/processed/{preset}/` e `data/manifests/`

* `X_train.csv`, `y_train.csv` (e equivalentes para val/test)
* `X_train.npy`, `y_train.npy` (e equivalentes para val/test)
* `manifest.json`


## Fluxo de Integração

                       ┌───────────────────────┐
                       │   extrair_dados.py    │
                       │   (CSV bruto)         │
                       └─────────────┬─────────┘
                                     │
                                     ▼
                       ┌────────────────────────┐
                       │  transformar_dados.py  │
                       │  (CSV full features)   │
                       └─────────────┬──────────┘
                                     │
                                     ▼
                       ┌────────────────────────┐
                       │   preparar_dados.py    │
                       │ (splits, normalização, │
                       │  janelas, artefatos)   │
                       └─────────────┬──────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │   Treinamento do Modelo      │
                     │ (Transformer / arquitetura)  │
                     └──────────────────────────────┘


### **Resumo do Fluxo**

1. **Extração** → gera dados brutos (OHLCV).
2. **Transformação** → gera dataset completo com features + rótulos.
3. **Preparação** → seleciona features, normaliza, gera janelas e produz artefatos de treino.

Este documento resume de forma direta e técnica o funcionamento dos três scripts essenciais do pipeline de dados.
