
# TRANS-BINARY — Documento Oficial de Escopo do Projeto

## 1. Visão Geral

O `trans-binary` é um sistema modular e automatizado para operações em **opções binárias com inteligência artificial**, desenvolvido em múltiplos MVPs. Ele abrange todo o fluxo desde dados brutos de mercado até execução automatizada na corretora, com interface gráfica (GUI) própria e suporte tanto para conta **demo** como **real**.

O produto final será distribuído em um arquivo ZIP com a aplicação empacotada (“GUI + modelos + configs”), além de uma CLI alternativa para automação ou uso avançado.

---

## 2. Objetivo Geral

Construir um sistema que:

1. Transforma dados históricos OHLCV em features e rótulos,
2. Treina e calibra um modelo Transformer,
3. Implementa uma política de decisão por aprendizado por reforço,
4. Executa ordens 100% automáticas na corretora, com controle de risco,
5. Funciona em **conta demo e real**, com mesma lógica e estabilidade,
6. Oferece uma **GUI principal de operação** e uma CLI secundária.

---
## 3. Estrutura de Diretórios

```
trans-binary/
├─ README.md                      ← Guia de instalação e uso (GUI e CLI)
├─ LICENSE                       
├─ docs/
│  ├─ SCOPE.md                    ← Documento oficial de escopo
│  └─ ARCHITECTURE.md             ← Descrição técnica detalhada
├─ data/
│  ├─ raw/                        ← Dados OHLCV brutos
│  ├─ processed/                  ← CSVs gerados na transformação (MVP 1)
│  └─ manifests/                  ← Infos de normalização e splits (MVP 2)
├─ models/
│  ├─ transformer/                ← Checkpoints do modelo Transformer
│  └─ policy/                     ← Política de decisão por RL
├─ logs/
│  ├─ training/                   ← Logs do treino do modelo
│  ├─ execution/                  ← Logs de ordens enviadas e auditoria
│  └─ simulation/                 ← Logs de simulação do executor
├─ reports/
│  └─ summaries/                  ← Relatórios e gráficos finais
├─ scripts/                       ← Scripts executáveis do projeto
│  ├─ build_dataset.py            ← MVP 1 e 2
│  ├─ train_transformer.py        ← MVP 3
│  ├─ calibrate_model.py          ← MVP 4
│  ├─ train_policy.py             ← MVP 5
│  ├─ run_executor.py             ← MVP 6 (modo CLI)
│  └─ build_gui.sh                ← Empacotamento da GUI
├─ src/
│  ├─ config/                     ← YAMLs (demo/real, hiperparâmetros)
│  ├─ transformacao_completa/     ← MVP 1 (features e labels)
│  ├─ preparacao/                 ← MVP 2 (normalização e splits)
│  ├─ model/                      ← MVP 3 e 4 (Transformer e calibração)
│  ├─ rl/                         ← MVP 5 (política de reforço)
│  ├─ modos/                      ← Controle de operação (Demo e Real)
│  ├─ executor/                   ← MVP 6 (API corretora, risco, auditoria)
│  ├─ gui/                        ← GUI da aplicação (PySide6)
│  └─ utils/                      ← E/S, métricas, manipuladores úteis
├─ tests/
└─ requirements.txt               ← Dependências de execução
```

---

## 4. Legenda de Arquivos

| Arquivo                           | Função                                                  |
| --------------------------------- | ------------------------------------------------------- |
| `README.md`                       | Instruções de instalação e uso                          |
| `docs/SCOPE.md`                   | Escopo completo do projeto                              |
| `data/raw/*.csv`                  | Dados OHLCV de entrada                                  |
| `data/processed/*.csv`            | Arquivos de features transformados                      |
| `data/manifests/*.json`           | Metainformações de normalização e splits                |
| `models/transformer/model.pt`     | Modelo base treinado                                    |
| `models/transformer/model_cal.pt` | Modelo calibrado                                        |
| `models/policy/policy.pt`         | Política de decisão por reforço                         |
| `scripts/build_dataset.py`        | Gera CSV formatado a partir de OHLCV                    |
| `scripts/train_transformer.py`    | Treina o modelo Transformer                             |
| `scripts/calibrate_model.py`      | Calibra modelo com Platt ou isotônica                   |
| `scripts/train_policy.py`         | Treina a política de reforço                            |
| `scripts/run_executor.py`         | Executa decisões do modelo pela CLI                     |
| `src/gui/`                        | Interface gráfica do app para operação demo/real        |
| `src/executor/runner.py`          | Integração com corretora, risco e ordens                |
| `src/modos/demo.yaml`             | Configuração de operação em conta demo                  |
| `src/modos/real.yaml`             | Configuração de operação em conta real                  |
| `src/preparacao/prepare.py`       | Script de normalização e splits temporais               |
| `src/utils/`                      | Código utilitário para IO, métricas, sincronização etc. |
| `requirements.txt`                | Módulos Python usados no projeto                        |

---

## 5. Dependências e Funções

| Biblioteca            | Finalidade                          |
| --------------------- | ----------------------------------- |
| pandas, numpy         | Manipulação de dados                |
| pytorch               | Modelo Transformer e política de RL |
| scikit-learn          | Calibração + métricas               |
| stable-baselines3     | Treino da política de reforço       |
| matplotlib            | Gráficos e relatórios               |
| PySide6               | Interface gráfica (GUI principal)   |
| requests / websockets | Integração com corretora            |
| pyinstaller           | Empacotamento final da GUI          |
| pyyaml                | Leitura de configurações (YAML)     |

Ambiente virtual: **virtualenv**
IDE de referência: **VSCode**
Controle de versão: **GitHub** com branches + tags por MVP

---
## 6. Arquitetura por Fases

```
[OHLCV bruto]
      │
      ▼
[MVP 1 — Transformação Completa] → CSV com features e labels
      │
      ▼
[MVP 2 — Preparação] → normalização e splits temporais
      │
      ▼
[MVP 3 — Modelo Transformer] → previsões no vencimento
      │
      ▼
[MVP 4 — Calibração] → modelo calibrado e validado por walk-forward
      │
      ▼
[MVP 5 — Política de Reforço] → decide CALL, PUT ou NÃO OPERAR
      │
      ▼
[Modos de Operação: Demo e Real] → configuração da corretora e risco
      │
      ▼
[MVP 6 — Executor + GUI] → envio automatizado de ordens + ZIP final
```
---


## 7. Funcionalidades por MVP

| MVP | Escopo                              | Entregáveis                                    |
| --- | ----------------------------------- | ---------------------------------------------- |
| 1   | Transformação completa de OHLCV     | CSV final com todas as features + labels       |
| 2   | Normalização e split                | `X_train`, `X_val`, `X_test` + manifestos JSON |
| 3   | Treino do modelo Transformer        | `model.pt` + logs de aprendizado               |
| 4   | Calibração e validação walk-forward | `model_calibrated.pt` + relatório              |
| 5   | Política de decisão por RL          | `policy.pt` + métricas de decisão              |
| —   | **Modos de Operação (Demo e Real)** | Configs de corretora e risco                   |
| 6   | Executor + GUI + Empacotamento      | ZIP final completo + CLI opcional              |

---

## 8. Critérios de Aceitação por MVP

Cada MVP só é encerrado após satisfazer:

* **MVP 1**: CSV completo sem vazamento temporal, com labels corretos.
* **MVP 2**: Splits temporais consistentes + normalização reversível.
* **MVP 3**: Modelo Transformer treinado com métricas mínimas aceitas.
* **MVP 4**: Probabilidades calibradas com melhora de confiabilidade (Brier).
* **MVP 5**: Política com estabilidade e EV positivo em validação.
* **MVP 6**: GUI funcional + executor real/demo + pacote ZIP validado.

---

## 9. Releases, Branches e Tags

| Release | Branch                | Tag           | Conteúdo                  |
| ------- | --------------------- | ------------- | ------------------------- |
| MVP 1   | `MVP-1/transformacao` | `v1.0.0-mvp1` | CSV final                 |
| MVP 2   | `MVP-2/preparacao`    | `v1.1.0-mvp2` | Normalização e splits     |
| MVP 3   | `MVP-3/transformer`   | `v1.2.0-mvp3` | Modelo Transformer        |
| MVP 4   | `MVP-4/calibracao`    | `v1.3.0-mvp4` | Modelo calibrado          |
| MVP 5   | `MVP-5/reforco`       | `v1.4.0-mvp5` | Política de reforço       |
| MVP 6   | `MVP-6/executor-gui`  | `v1.5.0-mvp6` | Executor, GUI e ZIP final |


---

## 10. Modos de Operação (entre MVP 5 e MVP 6)

| Modo              | Descrição                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| **Demo**          | Usa API ou interface da corretora demo, opera com saldo virtual, ideal para testes e backtests em tempo real. |
| **Real**          | Usa conta real com saldo financeiro. Mesma lógica, mesmas validações e mesmos proteções de risco.             |
| **Configurações** | Usuário define no GUI ou YAML: stake (mínimo 2), par, expiração, limite de perda diária e sequência.          |

---

## 11. Executor + GUI (MVP 6)

### Executor

* Recebe decisões (`CALL`, `PUT`, `NÃO OPERAR`) da política.
* Verifica pré-condições: ativo, payout mínimo, horário, risco.
* Envia ordens para a corretora (API ou automação GUI).
* Lida com **consequências reais ou demo**, baseado no modo configurado.
* Registra tudo com `client_order_id` para auditoria.
* Aplica risco: limite diário, perdas consecutivas, cooldown.

### GUI (PySide6)

* Interface principal do sistema.
* Abas: Dados, Modelo(s), Operação, Risco, Logs.
* Permite ao usuário:

  * Carregar modelo calibrado e política,
  * Selecionar modo (demo / real),
  * Configurar stake (>=2), par, expiração,
  * Iniciar e parar operação,
  * Acompanhar logs e métricas.


---

### Empacotamento

* GUI empacotada com `pyinstaller` em ZIP final.
* Alternativa: CLI (`run_executor.py`) para uso headless.

---

## 12. Uso Básico Após Instalado

```
unzip trans-binary-v1.5.0.zip
cd trans-binary
./transbinary_gui   # ou .exe

```

Na GUI:

1. Selecione **Modo Demo**,
2. Escolha modelo e política,
3. Defina stake >= 2 e par desejado,
4. Inicie operação.

---

## 13. Fluxo de Desenvolvimento (Orientativo)

```
Planejamento
│
├─ Criação do repositório no GitHub (branch main)
│
├─ MVP 1 (branch: MVP-1/transformacao)
│   └─ Transformação e labels → v1.0.0-mvp1
│
├─ MVP 2 (branch: MVP-2/preparacao)
│   └─ Normalização e splits → v1.1.0-mvp2
│
├─ MVP 3 (branch: MVP-3/transformer)
│   └─ Treino do modelo Transformer → v1.2.0-mvp3
│
├─ MVP 4 (branch: MVP-4/calibracao)
│   └─ Calibração e validação → v1.3.0-mvp4
│
├─ MVP 5 (branch: MVP-5/reforco)
│   └─ Política RL → v1.4.0-mvp5
│
└─ MVP 6 (branch: MVP-6/executor-gui)
    └─ Executor, GUI e pacote final → v1.5.0-mvp6
```

Cada MVP só é integrado à `main` após atender seus critérios de aceitação.

Responsável Carolina Brescowitt
