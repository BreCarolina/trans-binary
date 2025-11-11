# **TRANS-BINARY — Documento Oficial de Escopo do Projeto**

## **1. Visão Geral**

O **`trans-binary`** é um sistema preditivo inteligente voltado para **operações binárias**, concebido para **analisar, prever e agir automaticamente** no mercado financeiro.
Seu núcleo é baseado em **modelos Transformer** e **aprendizado por reforço**, permitindo a criação de um agente capaz de **aprender com o tempo**, **ajustar suas estratégias** e **tomar decisões autônomas**.

O projeto une **engenharia de dados, inteligência artificial e automação operacional**, integrando coleta de dados, processamento, modelagem, execução e interface em um único ecossistema.

O sistema será compatível com **modo Demo e Real**, incluirá **interface gráfica (GUI)** e **interface de linha de comando (CLI)**, e manterá **registro completo de logs e métricas** para rastreabilidade e análise de desempenho.

---

## **2. Objetivo Geral**

Desenvolver um **sistema preditivo autônomo** com base em aprendizado profundo e reforço contínuo, capaz de:

1. **Prever o próximo candle** (ou sequência de candles) a partir de dados históricos OHLCV;
2. **Adaptar-se dinamicamente** aos resultados obtidos, aprendendo com o próprio desempenho;
3. **Executar operações automáticas** de CALL / PUT com gestão de risco e confiança probabilística;
4. **Permitir operação em conta demo ou real**, com mesma estrutura e lógica;
5. **Registrar cada decisão e resultado** com métricas e histórico detalhado;
6. **Disponibilizar interface intuitiva** e uma CLI técnica para controle total das execuções.

---

## **3. Escopo Geral**

O projeto abrangerá todas as etapas necessárias para criar um sistema preditivo operacional completo:

* **Extração de dados** OHLCV diretamente da corretora IQ Option;
* **Transformação e enriquecimento dos dados**, com engenharia de features técnicas e temporais;
* **Normalização e divisão temporal (train/val/test)** sem vazamento de informação;
* **Treinamento supervisionado** do modelo Transformer (baseline e versões evolutivas);
* **Calibração probabilística e ajuste de confiança**;
* **Treinamento de política de decisão** via aprendizado por reforço;
* **Executor automatizado** de ordens em tempo real;
* **Interface GUI e CLI integradas** para operação humana e técnica;
* **Infraestrutura de logs, relatórios e versionamento completo.**

O primeiro protótipo terá como par principal **EURJPY** no timeframe **M5**, com suporte planejado a novos pares e intervalos.

---

## **4. Estrutura de Diretórios**

```
trans-binary/
├─ docs/
│  ├─ SCOPE.md                  ← Documento de escopo do projeto
│  └─ ARCHITECTURE.md           ← Detalhamento técnico do sistema
├─ data/
│  ├─ raw/                      ← Dados OHLCV brutos extraídos da IQ Option
│  ├─ processed/                ← Dados transformados e normalizados
│  └─ manifests/                ← Metadados e estatísticas de normalização
├─ models/
│  ├─ transformer/              ← Modelos preditivos (checkpoints)
│  └─ policy/                   ← Políticas de decisão (reforço)
├─ logs/
│  ├─ training/                 ← Logs de treino e validação
│  ├─ execution/                ← Logs de ordens e resultados
│  └─ simulation/               ← Logs de simulações e testes
├─ reports/
│  └─ summaries/                ← Relatórios e métricas de desempenho
├─ scripts/
│  ├─ extrair_dados.py          ← Coleta de candles OHLCV
│  ├─ transformar_dados.py      ← Criação e enriquecimento de features
│  ├─ preparar_dados.py         ← Normalização e divisão temporal
│  ├─ sequenciar_dados.py       ← Criação das janelas para o Transformer
│  ├─ treinar_transformer.py    ← Treinamento e validação do modelo
│  ├─ calibrate_model.py        ← Calibração de probabilidade e ajuste
│  ├─ train_policy.py           ← Aprendizado por reforço
│  ├─ run_executor.py           ← Executor CLI (execução autônoma)
│  └─ build_gui.sh              ← Empacotamento da interface gráfica
├─ src/
│  ├─ config/                   ← Arquivos YAML de configuração
│  ├─ transformacao_completa/   ← Funções auxiliares de feature engineering
│  ├─ preparacao/               ← Normalização e escalonamento
│  ├─ model/                    ← Implementação do Transformer
│  ├─ rl/                       ← Aprendizado por reforço e política
│  ├─ executor/                 ← Módulo de execução automática
│  ├─ gui/                      ← Interface PySide6
│  └─ utils/                    ← Funções auxiliares gerais
└─ requirements.txt
```

---

## **5. Tecnologias e Ferramentas**

| Tecnologia                   | Função                                    |
| ---------------------------- | ----------------------------------------- |
| **Python 3.12 + virtualenv** | Ambiente base de desenvolvimento          |
| **pandas / numpy**           | Manipulação e análise de séries temporais |
| **PyTorch**                  | Implementação do modelo Transformer       |
| **scikit-learn**             | Normalização e métricas                   |
| **stable-baselines3**        | Framework de aprendizado por reforço      |
| **matplotlib**               | Geração de relatórios e visualizações     |
| **PySide6**                  | Interface gráfica                         |
| **requests / websockets**    | Comunicação com a IQ Option               |
| **pyyaml**                   | Leitura de configurações YAML             |
| **pyinstaller**              | Empacotamento final executável            |
| **Git + GitHub**             | Versionamento e rastreabilidade de fases  |

---

## **6. Arquitetura Geral**

O sistema será desenvolvido em **fases interligadas**, formando um pipeline de aprendizado completo:

```
[Extração de Dados]
        ↓
[Transformação e Criação de Features]
        ↓
[Normalização e Splits Temporais]
        ↓
[Treinamento do Modelo Transformer]
        ↓
[Calibração e Ajuste de Confiança]
        ↓
[Aprendizado por Reforço]
        ↓
[Executor + GUI]
```

Cada etapa gera artefatos (CSV, manifestos, modelos, logs) usados como entrada da fase seguinte.
O fluxo garante **modularidade, rastreabilidade e reprodutibilidade total**.

---

## **7. Engenharia de Dados e Features**

A etapa de transformação converte os candles OHLCV brutos em representações preditivas com alto poder informativo.
Todas as features são calculadas **sem vazamento temporal**, respeitando a ordem cronológica dos dados.

### **Famílias de Features**

| Família                  | Exemplos                                     | Propósito                                 |
| ------------------------ | -------------------------------------------- | ----------------------------------------- |
| **Básicas (OHLCV)**      | abertura, máxima, mínima, fechamento, volume | Captura direta da estrutura de preço      |
| **Retornos e variações** | ret_log_1, ret_pct_1, ret_pct_5, ret_pct_12  | Mede tendência e intensidade de variação  |
| **Estrutura de candles** | body, wicks, range_pct_close                 | Identifica padrões de força ou reversão   |
| **Indicadores técnicos** | ema_7, ema_21, ema_50, rsi_14, atr_14        | Contexto de tendência e volatilidade      |
| **Estatísticas móveis**  | roll_mean_5, roll_std_12, zscore_48          | Normaliza o preço em relação ao histórico |
| **Codificação temporal** | hour_sin/cos, dow_sin/cos                    | Reconhece padrões intradiários e semanais |

Essas features são criadas em **`transformar_dados.py`** e armazenadas em `data/processed/`.

---

## **8. Normalização e Splits**

O script **`preparar_dados.py`** lê o dataset transformado e gera divisões temporais (train, val, test), aplicando normalização **z-score** apenas sobre o conjunto de treino.

Suporta diferentes **presets de features**:

| Preset             | Descrição                                                |
| ------------------ | -------------------------------------------------------- |
| **baseline_basic** | Apenas OHLCV + codificação temporal (para linha de base) |
| **tech_full**      | Todas as features técnicas e estatísticas                |
| **custom**         | Conjunto manual definido via YAML                        |

Essa estrutura permite testar o impacto incremental de cada grupo de features no desempenho do modelo.

---

## **9. Modelo Transformer**

O núcleo preditivo do sistema é um **Transformer Encoder** com **positional encoding aprendível** e **camadas totalmente conectadas** para classificação multiclasse.

### **Arquitetura Base**

* Projeção linear de entrada (`n_features → d_model`)
* `N` blocos Transformer Encoder (multi-head attention)
* Pooling médio temporal
* Normalização + camada de classificação
* Saída: logits para 3 classes (queda, estável, alta)

O modelo é configurável via YAML, com parâmetros como `d_model`, `nhead`, `num_layers`, `dropout` e `activation`.

---

## **10. Treinamento e Baseline**

A **baseline inicial** utilizará apenas **features básicas + temporais**, com o objetivo de avaliar a capacidade mínima do modelo de capturar direções de candle.

| Parâmetro    | Valor padrão |
| ------------ | ------------ |
| `seq_len`    | 64           |
| `horizon`    | 1            |
| `batch_size` | 256          |
| `lr`         | 1.5e-4       |
| `max_epochs` | 30           |
| `device`     | auto         |

Os resultados (acurácia, loss, curvas de validação e logs) são armazenados em:

* `logs/training/`
* `reports/summaries/`
* `models/transformer/`

Essa baseline será o **ponto de partida comparativo** para versões posteriores com features técnicas adicionais.

---

## **11. Executor e Interface**

O executor será responsável por **tomar decisões automáticas** com base nas previsões do modelo.

### **Funções Principais**

* Interpretar sinais do modelo (CALL / PUT / IGNORAR)
* Validar payout e risco
* Enviar ordem via API
* Registrar resultado, confiança e retorno financeiro
* Realimentar dados ao módulo de aprendizado por reforço

### **Interface GUI**

* Desenvolvida em **PySide6**
* Exibe:

  * Status da conexão e modo ativo (demo/real)
  * Confiabilidade do modelo
  * Histórico e resultados
* Controles:

  * Iniciar/parar operação
  * Selecionar par e stake
  * Alternar modo Demo/Real

### **CLI**

* Alternativa sem interface gráfica
* Permite automação e execução em servidores

---

## **12. Critérios de Aceitação**

O projeto será considerado funcional quando:

1. O modelo Transformer baseline atingir convergência estável;
2. A normalização e os splits preservarem integridade temporal;
3. O pipeline completo puder ser reproduzido com um único comando;
4. O executor operar em modo demo sem intervenção manual;
5. A interface exibir informações em tempo real;
6. Todos os logs e relatórios forem armazenados de forma organizada.

---

## **13. Versionamento e Controle**

| Fase          | Branch                 | Tag      | Artefato                      |
| ------------- | ---------------------- | -------- | ----------------------------- |
| Transformação | `fase-1/transformacao` | `v0.1.0` | Dataset de features           |
| Preparação    | `fase-2/preparacao`    | `v0.2.0` | Dados normalizados e splits   |
| Modelo        | `fase-3/modelo`        | `v0.3.0` | Transformer baseline          |
| Calibração    | `fase-4/calibracao`    | `v0.4.0` | Modelo calibrado              |
| Reforço       | `fase-5/reforco`       | `v0.5.0` | Política RL                   |
| GUI Final     | `fase-6/gui`           | `v1.0.0` | Executor + Interface completa |

---

## **14. Responsabilidade Técnica**

**Carolina Brescowitt**
Cientista de Dados — Projeto `trans-binary`
Foco: estrutura lógica, clareza técnica e evolução progressiva.
“Transformar complexidade em ordem e aprendizado contínuo.”

