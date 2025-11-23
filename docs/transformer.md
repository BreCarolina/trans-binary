
# **Documento – Modelo Transformer**

## **1. Modelo Transformer**

O modelo implementado em
`src/transformer/arquitetura.py`
é um **Transformer Encoder** adaptado para classificação binária de séries temporais financeiras.

### **Arquitetura**

* **Input:** `(batch, seq_len, n_features)`
* **Camada de projeção:**
  Converte `n_features → d_model` via `nn.Linear`.
* **Positional Encoding aprendível:**
  Embeddings aprendíveis com ajuste automático do comprimento da sequência.
* **Encoder Transformer:**
  Múltiplas camadas `nn.TransformerEncoderLayer`, com:

  * `d_model`
  * `n_heads`
  * `dim_feedforward`
  * `dropout`
  * `activation="gelu"`
  * `batch_first=True`
* **Pooling:**
  Estratégia configurável (`mean`, `max`, `cls`) para condensar a sequência.
* **Classificador final (MLP):**

  * `LayerNorm(d_model)`
  * `Linear → GELU → Dropout → Linear`
* **Saída:** `(batch, n_classes)`
  com logits para classificação binária.

---

## **2. Configuração Externa (YAML)**

O arquivo YAML define hiperparâmetros, caminhos e presets.

**Exemplo (transformer_baseline.yaml):**

```
preset: baseline_basic

feature_presets:
  baseline_basic:
    train_X: data/..._X_train.npy
    train_y: data/..._y_train.npy
    val_X:   data/..._X_val.npy
    val_y:   data/..._y_val.npy
    test_X:  data/..._X_test.npy
    test_y:  data/..._y_test.npy

seq_len: 36
n_features: 36
d_model: 96
n_heads: 4
num_layers: 2
dim_feedforward: 192
dropout: 0.1
pooling: mean
n_classes: 2

batch_size: 128
epochs: 5
learning_rate: 5e-4
weight_decay: 1e-6
device: auto

scheduler:
  type: ReduceLROnPlateau
  factor: 0.5
  patience: 3

paths:
  checkpoints_dir: models/transformer/checkpoints
  log_json: logs/training/train_transformer_log.json

save_every_n_epochs: 1
max_checkpoints: 10
```

### **Funções da Configuração**

* Define hiperparâmetros do modelo
* Define hiperparâmetros de otimização
* Fornece caminhos para os dados `.npy`
* Controla pooling, tamanho da janela e arquitetura interna
* Determina onde checkpoints e logs são salvos

O script de treinamento **infere automaticamente** `seq_len` e `n_features` a partir dos dados, sobrescrevendo valores do YAML quando necessário.

---

## **3. Treinamento**

O treinamento é realizado pelo arquivo:
`src/transformer/treinamento.py`

### **Processo**

1. Carrega o YAML
2. Aplica o preset e define automaticamente os caminhos de treino/val/test
3. Lê o shape do `train_X.npy` e infere:

   * `seq_len`
   * `n_features`
4. Instancia o modelo com `build_model_from_config(cfg)`
5. Cria DataLoaders
6. Loop de treinamento:

   * forward
   * loss (CrossEntropy)
   * backward
   * optimizer.step
   * scheduler opcional
7. Avaliação por época
8. Salvamento de checkpoints
9. Geração de log JSON
10. Avaliação final no conjunto de teste

---

## **4. Métricas**

As métricas acompanhadas durante o treinamento são:

### **Por época**

* **train_loss** — perda média no treino
* **val_loss** — perda média na validação
* **val_acc** — acurácia na validação
* **lr** — learning rate atual
* **epoch_time_s** — tempo da época

### **Ao final**

* **Acurácia no conjunto de teste**
  Exemplo de saída:

```
[TEST] Acurácia final: 0.8123 (823/1013)
```

### **Logs**

Armazenados em:

```
logs/training/train_transformer_log.json
```

### **Checkpoints**

Salvos em:

```
models/transformer/checkpoints/
model_epoch_001.pth
...
```

---

## **5. Resumo**

Este documento descreve de forma objetiva:

* A arquitetura do modelo Transformer
* Os parâmetros externos controlados via YAML
* O processo de treinamento
* As métricas monitoradas
* Os artefatos gerados (checkpoints e logs)

O modelo está totalmente modular, configurável e otimizado para janelas temporais financeiras.

