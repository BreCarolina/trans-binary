# **Fluxo Operacional — Projeto trans-binary**


[INÍCIO]
   |
   v
──────────────────────────────────────────────────────────────
| 0. AMBIENTE & CREDENCIAIS                                   |
| - Ambiente virtual (.venv, Python 3.12)                     |
| - Arquivo .env com: IQ_EMAIL, IQ_PASSWORD                   |
| - requirements.txt (numpy<2, pandas, torch, etc.)           |
| - Estrutura trans-binary/ criada e versionada               |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 1. EXTRAÇÃO DE HISTÓRICOS (scripts/extrair_dados.py)        |
| - Carrega credenciais do .env                               |
| - Conecta à IQ Option (login automático)                    |
| - Converte timeframe textual → segundos (ex: 'm5'→300s)     |
| - Define cursor inicial (último candle fechado)             |
| - Loop de busca reversa em blocos (máx. 1000 candles/call): |
|     * api.get_candles(par, timeframe, qtd, cursor)          |
|     * concatena blocos; atualiza cursor                     |
| - Normaliza colunas → [from, abertura, maxima, minima,      |
|                        fechamento, volume]                  |
| - Remove candles incompletos                                |
| - Filtra horário líquido 07:00–21:00 UTC                    |
| - Se CSV já existir → mescla incremental (dedup por 'from') |
| - Salva em data/raw/{PAR}_M{TF}_{dias}d.csv                 |
| - Loga caminho salvo e tail() do DataFrame                  |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 1.1 CONTROLE DE QUALIDADE DA EXTRAÇÃO                       |
| - Checa não-vazio após filtro horário                       |
| - Confirma ordenação temporal ascendente                    |
| - Elimina duplicatas por timestamp                          |
| - Valida tipos numéricos e intervalos                       |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 2. TRANSFORMAÇÃO E FEATURES (scripts/transformar_dados.py)  |
| - Localiza o CSV bruto mais recente (par/timeframe)         |
| - Carrega OHLCV ordenado, tipado e sem duplicatas           |
| - Calcula features (sem vazamento temporal):                 |
|     * feat_returns_close_log() — retornos e variações        |
|     * feat_body_wicks() — estrutura do candle                |
|     * feat_ema_cross() — médias móveis e cruzamentos         |
|     * feat_rsi(), feat_atr() — força e volatilidade          |
|     * feat_rolling_stats() — médias/std/zscore móveis        |
|     * feat_time_cyclical() — hora/dia codificados seno/cos   |
| - Cria labels com base no próximo fechamento:                |
|     * y_up, y_down, y_noop (sem vazamento)                   |
| - Concatena base + X + Y; dropa NaNs iniciais e última linha |
| - Salva em data/processed/{PAR}_M{TF}_features_{N}x.csv     |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 2.1 QC DAS FEATURES                                         |
| - Checa presença de NaNs                                    |
| - Verifica consistência dos rótulos (one-hot soma=1)        |
| - Confere monotonicidade temporal                           |
| - Gera relatório: reports/summaries/{PAR}_M{TF}_qc.md       |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 3. PREPARAÇÃO (scripts/preparar_dados.py)                   |
| - Lê o arquivo features_{N}x.csv                            |
| - Ordena por tempo; define cortes 70/15/15 (train/val/test) |
| - Ajusta preset de features (baseline, tech_full, custom)   |
| - FIT do scaler (StandardScaler) somente no treino          |
| - Aplica scaler nos conjuntos de validação e teste          |
| - Salva splits: X/y train, val, test (CSV)                  |
| - Gera manifesto JSON com:                                  |
|     * colunas, mean/std, índices de corte                   |
|     * ratios e datas de cada split                          |
|     * caminho de origem e hash opcional                     |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 3.1 QC DA PREPARAÇÃO                                        |
| - Recarrega manifesto e splits                              |
| - Verifica correspondência das colunas                      |
| - Confere shapes e consistência do scaler                   |
| - Reporta “OK” ou falhas encontradas                        |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 4. SEQUENCIALIZAÇÃO (src/model/sequenciar_dados.py)         |
| - Lê config YAML (seq_len, horizon, target)                 |
| - Cria janelas deslizantes sem vazamento temporal           |
| - Gera:                                                     |
|     * X: [N, seq_len=64, n_features≈30]                     |
|     * y: [N,] → direção (0,1,2)                             |
| - Aplica target_map (direction ou class3)                   |
| - Imprime shapes e estatísticas básicas                     |
| - Retorna pacotes np.ndarray para treino                    |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 5. CONFIGURAÇÃO (src/config/eurjpy_m5.yaml)                 |
| - data: caminhos X/y train, val, test e manifesto            |
| - sequence: seq_len=64, horizon=1, target=direction          |
| - model: d_model, nhead, num_layers, dropout, activation     |
| - train: batch_size, lr, weight_decay, max_epochs, seed      |
| - paths: checkpoints_dir, logs_dir, outputs_dir              |
| - runtime: num_workers, pin_memory, device=auto              |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 6. ARQUITETURA (src/model/transformer.py)                   |
| - Projeção Linear (n_features → d_model)                    |
| - PositionalEncoding aprendível                             |
| - Encoder empilhado (Multi-Head Self-Attention + FFN)       |
| - Pooling temporal (mean)                                   |
| - LayerNorm + cabeça linear → logits (3 classes)            |
| - Loss: CrossEntropyLoss                                    |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 7. TREINAMENTO (scripts/treinar_transformer.py)             |
| - Define seed e device (auto/cpu/cuda)                      |
| - Cria DataLoaders (train/val/test)                         |
| - Otimizador: AdamW (lr, weight_decay, grad_clip)           |
| - Loop de épocas:                                           |
|     [TREINO]                                                |
|       1) forward(Xb) → logits                               |
|       2) loss = CE(logits, yb)                              |
|       3) backward() + grad clip + optimizer.step()          |
|       4) acumula métricas                                   |
|     [VALIDAÇÃO]                                             |
|       5) no_grad(); calcula val_loss/val_acc                |
|       6) se val_acc ↑ → salva checkpoint “best”             |
| - Ao final: carrega melhor .pt e roda teste final           |
| - Gera log JSON em logs/training/                           |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 8. SAÍDAS E ARTEFATOS                                       |
| - models/transformer/*_best.pt → checkpoint eleito          |
| - logs/training/*.json → histórico de treino                 |
| - reports/summaries/ → relatórios futuros                   |
| - data/manifests/*.json → referência para inferência        |
──────────────────────────────────────────────────────────────
   |
   v
──────────────────────────────────────────────────────────────
| 9. VERSIONAMENTO E RELEASES                                 |
| - Cada fase possui branch dedicada                          |
| - Fluxo: branch → validação → merge na main                 |
| - Tag semântica (v0.1.0, v0.2.0, …)                         |
| - Release no GitHub com notas resumidas                     |
──────────────────────────────────────────────────────────────
   |
   v
[FIM]
