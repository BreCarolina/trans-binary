
# TRANS-BINARY — Documento Oficial de Escopo do Projeto

## 1. Visão Geral

O `trans-binary` é um sistema preditivo inteligente voltado para **opções binárias**, desenvolvido para aprender e evoluir continuamente através de **modelos Transformer** e **aprendizado por reforço**.
Seu objetivo é prever o próximo (ou próximos) candles de pares de moedas e executar operações automáticas com base nas previsões, integrando **análise, decisão e execução** em um único ambiente.

O sistema será compatível com **modo demo e real**, oferecerá uma **interface gráfica intuitiva (GUI)** e uma **CLI** como alternativa de uso técnico. Todas as operações, resultados e métricas serão registrados em logs locais para rastreabilidade.

---

## 2. Objetivo Geral

Desenvolver um **modelo preditivo autônomo** com arquitetura Transformer e aprendizado por reforço contínuo, capaz de:

1. **Prever o próximo candle ou sequência de candles** a partir de dados OHLCV históricos;
2. **Ajustar-se e evoluir** com base nas operações executadas (reforço contínuo);
3. **Operar automaticamente** tanto em **conta demo** quanto em **conta real**, usando a mesma lógica;
4. **Permitir configuração de operação** (valor, par de moedas, expiração, modo demo/real);
5. **Registrar todas as operações**, incluindo entrada, horário, payout, resultado (win/loss), e confiança do modelo;
6. **Fornecer interface GUI intuitiva**, com indicadores de saúde do modelo e confiança das previsões;
7. Oferecer **CLI secundária** para uso avançado ou integração automatizada.

---

## 3. Escopo Geral

O projeto engloba as seguintes etapas principais:

* **Coleta e preparação de dados**: Extração e transformação de candles OHLCV.
* **Construção do modelo Transformer**: Treinamento supervisionado inicial.
* **Aprendizado por reforço**: Ajuste dinâmico com base em resultados reais das operações.
* **Executor de ordens**: Envio automático para a corretora com controle de risco.
* **Interface gráfica (GUI)**: Operação intuitiva e visual, exibindo status e métricas.
* **Empacotamento final**: Aplicativo ZIP executável, com GUI e CLI incluídas.

O primeiro MVP cobrirá o par **EURJPY** no timeframe **M5**, com previsão de expansão futura para novos pares e timeframes.

---

## 4. Estrutura de Diretórios

```
trans-binary/
├─ README.md
├─ LICENSE
├─ MVP/                          ← Produtos funcionais liberados (binários e configs)
├─ docs/
│  ├─ SCOPE.md                   ← Documento de escopo
│  └─ ARCHITECTURE.md            ← Detalhamento técnico
├─ data/
│  ├─ raw/                       ← Dados OHLCV brutos
│  ├─ processed/                 ← Dados transformados
│  └─ manifests/                 ← Metainformações (normalização e splits)
├─ models/
│  ├─ transformer/               ← Checkpoints do modelo preditivo
│  └─ policy/                    ← Política de decisão por reforço
├─ logs/
│  ├─ training/                  ← Logs de treino e ajuste do modelo
│  ├─ execution/                 ← Logs de ordens e resultados
│  └─ simulation/                ← Logs de simulação e teste
├─ reports/
│  └─ summaries/                 ← Relatórios e métricas
├─ scripts/
│  ├─ extrair_dados.py           ← Extração de candles OHLCV
│  ├─ build_dataset.py           ← Transformação e feature engineering
│  ├─ prepare_dataset.py         ← Normalização e splits temporais
│  ├─ train_transformer.py       ← Treinamento inicial do modelo
│  ├─ calibrate_model.py         ← Calibração e ajuste
│  ├─ train_policy.py            ← Treino da política de reforço
│  ├─ run_executor.py            ← Executor CLI
│  └─ build_gui.sh               ← Empacotamento da interface
├─ src/
│  ├─ config/                    ← YAMLs de configuração
│  ├─ transformacao_completa/    ← Código de transformação de dados
│  ├─ preparacao/                ← Código de normalização
│  ├─ model/                     ← Módulo Transformer
│  ├─ rl/                        ← Módulo de reforço
│  ├─ modos/                     ← Configurações demo/real
│  ├─ executor/                  ← Controle e execução de ordens
│  ├─ gui/                       ← Interface gráfica (PySide6)
│  └─ utils/                     ← Funções auxiliares e IO
├─ tests/
└─ requirements.txt
```

---

## 5. Tecnologias e Dependências

| Biblioteca            | Função principal                                  |
| --------------------- | ------------------------------------------------- |
| pandas, numpy         | Manipulação e cálculo de séries temporais         |
| pytorch               | Implementação do Transformer e reforço            |
| scikit-learn          | Calibração, métricas e normalização               |
| stable-baselines3     | Base de aprendizado por reforço (policy training) |
| matplotlib            | Geração de relatórios e gráficos                  |
| PySide6               | Interface gráfica do sistema                      |
| requests / websockets | Comunicação com a corretora                       |
| pyinstaller           | Empacotamento em executável                       |
| pyyaml                | Leitura de parâmetros e configurações             |

Ambiente virtual: **virtualenv**
IDE de referência: **VSCode**
Controle de versão: **GitHub (branches e tags por fase)**

---

## 6. Arquitetura por Fases

```
[Fase 0 — Extração de Dados]
        ↓
[Fase 1 — Transformação e Criação de Features]
        ↓
[Fase 2 — Normalização e Splits Temporais]
        ↓
[Fase 3 — Treinamento do Modelo Transformer]
        ↓
[Fase 4 — Calibração e Validação]
        ↓
[Fase 5 — Aprendizado por Reforço e Política de Decisão]
        ↓
[Fase 6 — Executor e GUI Final]
```

---

## 7. Releases e Versionamento

| Release | Branch                 | Tag      | Conteúdo                       |
| ------- | ---------------------- | -------- | ------------------------------ |
| Fase 1  | `fase-1/transformacao` | `v0.1.0` | Dataset pronto                 |
| Fase 2  | `fase-2/preparacao`    | `v0.2.0` | Normalização e splits          |
| Fase 3  | `fase-3/modelo`        | `v0.3.0` | Transformer treinado           |
| Fase 4  | `fase-4/calibracao`    | `v0.4.0` | Modelo calibrado               |
| Fase 5  | `fase-5/reforco`       | `v0.5.0` | Política RL                    |
| Fase 6  | `fase-6/gui`           | `v1.0.0` | GUI + Executor (produto final) |

---

## 8. Modos de Operação

| Modo     | Descrição                                                                         |
| -------- | --------------------------------------------------------------------------------- |
| **Demo** | Usa conta virtual, sem risco financeiro real. Ideal para testes e aprendizado.    |
| **Real** | Executa em conta real com stake definido pelo usuário, aplicando limites e risco. |

Ambos os modos compartilham a mesma base lógica e política de decisão.

---

## 9. Executor e GUI

### Executor

* Recebe sinais do modelo (CALL / PUT / NÃO OPERAR)
* Avalia condições de risco e payout mínimo
* Envia ordem via API ou automação
* Registra: data, hora, par, stake, resultado, payout, confiança, saúde do modelo

### GUI

* Interface principal (PySide6)
* Exibe:

  * Status de conexão e modo (Demo/Real)
  * Saúde do modelo e confiança
  * Resultados recentes
  * Logs de operação
* Botões principais:

  * Iniciar / Parar operação
  * Selecionar par e valor da stake
  * Alternar entre modo Demo e Real

### CLI

Alternativa headless com mesmos parâmetros.

---

## 12. MVP — Produto Funcional

### 12.1 Fluxograma 

```
[Início]
   |
   v
[Carregar Modelo Calibrado]
   |
   v
[Carregar Política de Reforço]
   |
   v
[Selecionar Modo: Demo ou Real]
   |
   v
[Definir Par (EURJPY) e Valor da Operação]
   |
   v
[Iniciar Execução]
   |
   v
+--------------------------------------+
| Loop Operacional                     |
| - Obter candle atual                 |
| - Prever próximo candle              |
| - Decidir ação (CALL / PUT / NONE)   |
| - Enviar ordem                       |
| - Registrar resultado e aprendizado  |
+--------------------------------------+
   |
   v
[Parar Execução]
   |
   v
[Fim]
```

### 12.2 Critérios de Aceitação do MVP

* Modelo Transformer funcional e calibrado.
* Política de reforço ativa e aprendendo com operações.
* Execução automatizada no par **EURJPY M5**.
* GUI e CLI operacionais.
* Registro completo das execuções (entrada, resultado, payout, confiança, timestamp).
* Empacotamento ZIP funcional.

---

Desenvolvimento

│
├─ Fase 0 — Extração de Dados
│      └─ Coleta de candles OHLCV em CSV (par: EURJPY, timeframe: M5, histórico bruto)
│
├─ Fase 1 — Transformação e Rotulagem
│      └─ Features técnicas e temporais + labels sem vazamento temporal (CSV final)
│
├─ Fase 2 — Normalização e Splits
│      └─ Divisão temporal em train/val/test + manifesto de normalização (JSON)
│
├─ Fase 3 — Treinamento Transformer
│      └─ Treino inicial do modelo Transformer → `model.pt` + logs e métricas
│
├─ Fase 4 — Calibração do Modelo
│      └─ Ajuste de probabilidade (Platt / Isotônica) → `model_cal.pt` + relatórios
│
├─ Fase 5 — Aprendizado por Reforço
│      └─ Política de decisão RL em constante aprendizado → `policy.pt`
│
└─ Fase 6 — Executor + GUI + Empacotamento
       └─ Executor real/demo + GUI operativa + empacotamento final (ZIP executável)

```

Cada fase é integrada à `main` após aprovação e validação funcional.

Responsável: **Carolina Brescowitt**

---

