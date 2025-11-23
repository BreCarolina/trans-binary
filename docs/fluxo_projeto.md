                    +---------------------------+
                    |     IQ Option / OHLCV     |
                    |     Fonte de Dados Brutos |
                    +-------------+-------------+
                                 |
                                 v
             ┌────────────────────────────────────────────┐
             │                   EXTRAÇÃO                 │
             │ Script: src/etl/extrair_dados.py           │
             │ Função: baixar candles OHLCV               │
             │ Artefato: data/raw/<PAR>_<TF>_<dias>.csv   │
             └────────────────────────────────────────────┘
                                  |
                                  v
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                                 TRANSFORMAÇÃO                            │
 │ Script: src/etl/transformar_dados.py                                     │
 │ Função: cálculo de TODAS as features (técnicas, estatísticas, temporais) │
 │ Artefato: data/processed/<PAR>_<TF>_full_<Nx>.csv                        │
 └──────────────────────────────────────────────────────────────────────────┘
                                    |
                                    v
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                          PREPARAÇÃO                                     │
 │ Script: src/etl/preparar_dados.py                                       │
 │ Função: seleção por preset, normalização, janelamento (seq_len), splits │
 │ Artefatos:                                                              │
 │   - CSVs normalizados (train/val/test)                                  │
 │   - .npy janelados em data/processed/<preset>/                          │
 │   - Manifesto JSON em data/manifests/                                   │
 └─────────────────────────────────────────────────────────────────────────┘
                                     |
                                     v
 ┌──────────────────────────────────────────────────────────────────────┐
 │                            ARQUITETURA DO MODELO                     │
 │ Módulo: src/transformer/arquitetura.py                               │
 │ Componentes:                                                         │
 │   • Linear (n_features → d_model)                                    |
 │   • Positional Encoding aprendível (interpolável)                    │
 │   • TransformerEncoder (n_layers, n_heads, FFN, GELU)                │
 │   • Pooling: mean / max / cls                                        │
 │   • Classifier final (LayerNorm → Linear → GELU → Dropout → Linear)  │
 │ Saída: logits (classes 0/1)                                          │
 └──────────────────────────────────────────────────────────────────────┘
                                    |
                                    v
      ┌───────────────────────────────────────────────────────────────┐
      │                    CONFIGURAÇÃO VIA YAML                      │
      │ Arquivo: src/transformer/config/transformer_baseline.yaml     │
      │ Conteúdos:                                                    │
      │   • Caminhos dos artefatos (.npy)                             │
      │   • Hiperparâmetros                                           │
      │   • Preset de features                                        │
      │   • Configuração de treinamento, scheduler, checkpoints       │
      └───────────────────────────────────────────────────────────────┘
                                    |
                                    v
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                           TREINAMENTO                                    │
 │ Script: src/transformer/treinamento.py                                   │
 │ Função: carregar .npy → montar dataloaders → instanciar modelo → treinar │
 │ Artefatos:                                                               │
 │   • Checkpoints .pth                                                     │
 │   • logs/training/*.json                                                 │
 │   • Métricas (loss, acc, LR, tempo/época)                                │
 └──────────────────────────────────────────────────────────────────────────┘
                                     |
                                     v
           ┌───────────────────────────────────────────────────────┐
           │                               MÉTRICAS / AVALIAÇÃO    │
           │ Acurácia final no conjunto de teste                   │
           │ Registros completos em reports/ e logs/training/      │
           └───────────────────────────────────────────────────────┘
                                     |
                                     v
                ┌────────────────────────────────────────────┐
                │                   MODELO FINAL             │
                │ Arquivos: checkpoints .pth + config usada  │
                │ Uso: carregado pelo Executor / GUI         │
                └────────────────────────────────────────────┘
                                    |
                                    v
                 ┌─────────────── FEEDBACK ────────────────┐
                 |                                         |                        
 ┌───────────────────────────────┐        ┌────────────────────────────────────┐     
 │              EXECUTOR         │        │         APRENDIZADO POR REFORÇO    │     
 │ src/executor/                 │◀──────▶│ Módulo RL (policy, rewards, buffer)│     
 │ - recebe previsão             │        │ - aprende com decisões reais       │    
 │ - aplica regras + confiança   │        │ - ajusta política continuamente    │   
 │ - coloca ordem na corretora   │        │ - retroalimenta Executor           │      
 │ - gera logs para RL e GUI     │        │                                    │   
 └───────────────────────────────┘        └────────────────────────────────────┘     
                 |                                           |
                 └───────────────────────────────────────────┘
                                     |
                                     v
            ┌───────────────────────────────────────────────────────┐
            │ GUI / MVP                                             │
            │ src/gui/   +   MVP/ (executável)                      │
            │ Funções:                                              │
            │   • visualizar modelo em tempo real                   │
            │   • acompanhar confiança, candles, sinais, histórico  │
            │   • modo Demo/Real                                    │
            │   • controle Start/Stop do Executor                   │
            └───────────────────────────────────────────────────────┘
