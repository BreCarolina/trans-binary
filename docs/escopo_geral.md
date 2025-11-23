
# **TRANS-BINARY — Documento Oficial de Escopo do Projeto**

## **1. Visão Geral**

O **`trans-binary`** é um sistema preditivo inteligente para **operações binárias**, projetado para **analisar, prever e agir automaticamente** no mercado financeiro.
Seu núcleo é baseado em **modelos Transformer**, com arquitetura modular que abrange:

* engenharia de dados completa,
* pipelines reprodutíveis,
* modelo preditivo supervisionado,
* executor operacional em tempo real,
* automação e interface.

O sistema oferecerá operação **Demo e Real**, interface **GUI** e **CLI**, além de logs detalhados para auditoria.

---

## **2. Objetivo Geral**

Construir um sistema capaz de:

1. Prever a direção do próximo candle usando deep learning.
2. Atualizar estratégias e operar com autonomia.
3. Gerir risco e confiança preditiva.
4. Executar operações CALL/PUT automaticamente.
5. Registrar decisões, métricas e histórico.
6. Oferecer controle via GUI e CLI.

---

## **3. Escopo Geral**

O projeto engloba:

* Extração de dados OHLCV da IQ Option
* Transformação e engenharia de features
* Normalização e splits temporais
* Treinamento supervisionado com Transformer
* Calibração e política de decisão
* Executor automatizado (demo/real)
* Interface gráfica
* Log, relatório e versionamento técnico

O protótipo inicial trabalha com **EURJPY — M5**, mas o sistema é expansível.

---

## **4. Estrutura de Diretórios**

```
trans-binary/
├── MVP/
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── baseline/
│   └── manifests/
├── docs/
├── logs/
│   ├── execution/
│   ├── simulation/
│   └── training/
├── models/
├── reports/
│   └── summaries/
├── src/
│   ├── etl/
│   ├── transformer/
│   │   └── config/
│   ├── tests/
│   ├── executor/
│   ├── gui/
│   └── utils/
├── .gitignore
├── README.md
└── requirements.txt
```

### **Legenda dos Diretórios**

| Diretório            | Descrição                                                  |
| -------------------- | ---------------------------------------------------------- |
| **MVP/**             | Onde ficará o executável final do protótipo (PyInstaller). |
| **data/raw/**        | Dados OHLCV brutos extraídos da IQ Option.                 |
| **data/processed/**  | Dados transformados, normalizados e janelados.             |
| **data/manifests/**  | Metadados e manifestos dos pipelines.                      |
| **docs/**            | Documentação técnica do projeto.                           |
| **logs/**            | Registros de execução, treinamento e simulação.            |
| **models/**          | Modelos treinados e checkpoints.                           |
| **reports/**         | Relatórios e análises geradas.                             |
| **src/etl/**         | Scripts de extração, transformação e preparação dos dados. |
| **src/transformer/** | Arquitetura, treinamento e configs do modelo Transformer.  |
| **src/tests/**       | Testes automatizados do pipeline.                          |
| **src/executor/**    | Executor automático de operações CALL/PUT.                 |
| **src/gui/**         | Interface gráfica PySide6.                                 |
| **src/utils/**       | Funções auxiliares, logs, utilidades.                      |

---

## **5. Tecnologias e Ferramentas**

| Tecnologia            | Função                           |
| --------------------- | -------------------------------- |
| Python 3.12           | Ambiente principal               |
| pandas / numpy        | Manipulação de dados             |
| PyTorch               | Modelo Transformer               |
| scikit-learn          | Normalização, métricas           |
| stable-baselines3     | Aprendizado por reforço (futuro) |
| PySide6               | Interface gráfica                |
| requests / websockets | Comunicação com IQ Option        |
| pyyaml                | Configurações externas via YAML  |
| pyinstaller           | Execução final empacotada        |
| Git + GitHub          | Versionamento                    |

---

## **6. Arquitetura Geral**

```
[Extração]
    ↓
[Transformação + Features]
    ↓
[Normalização + Splits + Janelas]
    ↓
[Modelo Transformer]
    ↓
[Calibração e Confiança]
    ↓
[Executor Automático]
    ↓
[GUI / CLI]
```

---

## **7. Engenharia de Dados**

O pipeline de dados possui três scripts principais:

### **extrair_dados.py**

* Conecta à API da IQ Option.
* Baixa candles OHLCV.
* Salva em `data/raw/`.

### **transformar_dados.py**

* Gera features avançadas (técnicas, estatísticas, temporais).
* Remove duplicações, ordena, limpa.
* Salva CSV completo em `data/processed/`.

### **preparar_dados.py**

* Seleciona features via preset.
* Cria splits temporais (train/val/test) sem vazamento.
* Normaliza somente com base no treino.
* Gera janelas (seq_len) para o Transformer.
* Salva CSVs, `.npy` e manifesto.

---

## **8. Modelo Transformer**

A arquitetura utiliza:

* Projeção linear de features
* Positional Encoding aprendível
* Transformer Encoder (camadas configuráveis)
* Pooling (mean / max / cls)
* Classificador final (direção binária)

Entrada: `(batch, seq_len, n_features)`
Saída: logits binários — **alta (1)** ou **baixa (0)**.

Configurações vêm de um arquivo YAML, incluindo:

* seq_len
* n_features
* d_model
* n_heads
* num_layers
* dropout
* caminhos dos datasets
* parâmetros de treinamento

---

## **9. Treinamento**

O script de treinamento:

* Lê o YAML
* Carrega os datasets janelados
* Infere automaticamente `seq_len` e `n_features`
* Cria DataLoaders
* Treina com AdamW
* Suporta scheduler ReduceLROnPlateau
* Salva checkpoints
* Registra logs a cada época
* Avalia o conjunto de teste ao final

Métricas de interesse:

* **train_loss**
* **val_loss**
* **val_acc**
* **test_acc**

---

## **10. Executor e Interface**

### Executor

* Recebe a previsão (0/1) do modelo.
* Valida payout, volatilidade e risco.
* Envia ordem CALL/PUT.
* Registra resultado e confiança.
* Alimenta a política futura de RL.

### GUI (PySide6)

* Status da conexão
* Confiança do modelo
* Histórico de operações
* Botões de iniciar/parar
* Seleção de stake, modo demo/real

### CLI

* Operação técnica, sem interface
* Usado em servidores

---

## **11. Critérios de Aceitação**

1. Pipeline reproduzível de ponta a ponta.
2. Splits temporais válidos.
3. Modelo baseline convergindo.
4. Executor operando em demo.
5. GUI funcionando em tempo real.
6. Logs organizados por fase.

---

## **12. Versionamento**

| Fase          | Branch | Tag    |
| ------------- | ------ | ------ |
| Extração      | fase-1 | v0.1.0 |
| Transformação | fase-2 | v0.2.0 |
| Preparação    | fase-3 | v0.3.0 |
| Modelo        | fase-4 | v0.4.0 |
| Executor      | fase-5 | v0.5.0 |
| GUI Final     | fase-6 | v1.0.0 |

---

## **13. Responsabilidade Técnica**

**Carolina Brescowitt**
Cientista de Dados — `trans-binary`
Foco: precisão, clareza e evolução progressiva.

