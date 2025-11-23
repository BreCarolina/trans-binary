#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
extract_val_outputs.py — versão corrigida para ajustar cfg a partir dos dados de validação
"""

import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any

from src.transformer.arquitetura import build_model_from_config

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Extrai logits, probs e labels do conjunto de validação.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--val-seqs', required=True)
    parser.add_argument('--val-labels', required=True)
    parser.add_argument('--out', required=False, default='models/transformer/calibration')
    parser.add_argument('--batch-size', required=False, type=int, default=256)
    parser.add_argument('--device', required=False, default=None)
    args = parser.parse_args()

    # carregar arrays de validação antes de construir o modelo
    val_seqs = np.load(args.val_seqs)
    val_labels = np.load(args.val_labels)

    if val_seqs.ndim != 3:
        raise ValueError('val_seqs must be shape (N, seq_len, n_features)')

    seq_len = int(val_seqs.shape[1])
    n_features = int(val_seqs.shape[2])

    # carregar cfg e ajustar parâmetros fundamentais a partir dos dados
    cfg = load_config(args.config)
    # sincroniza seq_len e n_features com os dados de validação (corrige YAML inconsistênte)
    cfg['seq_len'] = seq_len
    cfg['n_features'] = n_features

    _ensure_dir(args.out)

    # constrói modelo com cfg coerente
    model = build_model_from_config(cfg)

    # carregar checkpoint flexível
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    loaded = False
    if isinstance(ckpt, dict):
        # prefer model_state_dict, depois state_dict, depois fallback para ckpt inteiro
        for key in ('model_state_dict', 'state_dict'):
            if key in ckpt:
                state = ckpt[key]
                try:
                    model.load_state_dict(state)
                    loaded = True
                    break
                except Exception as e:
                    # tentar remover prefixos 'module.' automaticamente
                    new_state = {}
                    for k, v in state.items():
                        new_state[k.replace('module.', '')] = v
                    try:
                        model.load_state_dict(new_state)
                        loaded = True
                        break
                    except Exception as e2:
                        print(f'warn: tentativa com chave \"{key}\" falhou: {e}; fallback tentou strip module: {e2}')
        if not loaded:
            # tentar carregar ckpt diretamente (às vezes checkpoint é um state_dict sem wrapper)
            try:
                model.load_state_dict(ckpt)
                loaded = True
            except Exception as e:
                print('Erro ao carregar state_dict diretamente do checkpoint:', e)
    else:
        # checkpoint não é dict — tentar carregar direto
        try:
            model.load_state_dict(ckpt)
            loaded = True
        except Exception as e:
            print('Erro ao carregar checkpoint não-dict:', e)

    if not loaded:
        # falha ao carregar pesos — aborta com mensagem útil
        raise RuntimeError('Falha ao carregar pesos do checkpoint. Verifique compatibilidade entre cfg (seq_len/n_features) e checkpoint. Detalhes acima.')

    device = torch.device(args.device) if args.device else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

    model = model.to(device)
    model.eval()

    N = val_seqs.shape[0]
    batch = args.batch_size

    logits_out = []
    probs_out = []

    with torch.no_grad():
        for i in range(0, N, batch):
            x = val_seqs[i:i+batch]
            x = torch.from_numpy(x).float().to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            logits = out.detach().cpu()
            if logits.ndim == 2 and logits.shape[1] == 2:
                probs = F.softmax(logits, dim=1)[:, 1]
            elif logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                probs = torch.sigmoid(logits.view(-1))
            else:
                probs = F.softmax(logits, dim=-1)[:, -1]
            logits_out.append(logits.numpy())
            probs_out.append(probs.numpy())

    logits_all = np.concatenate(logits_out, axis=0)
    probs_all = np.concatenate(probs_out, axis=0)

    if logits_all.ndim == 2 and logits_all.shape[1] == 1:
        logits_all = logits_all.reshape(-1)

    np.save(os.path.join(args.out, 'val_logits.npy'), logits_all)
    np.save(os.path.join(args.out, 'val_probs.npy'), probs_all)
    np.save(os.path.join(args.out, 'val_labels.npy'), val_labels)

    print('Salvo com sucesso:')
    print(os.path.join(args.out, 'val_logits.npy'))
    print(os.path.join(args.out, 'val_probs.npy'))
    print(os.path.join(args.out, 'val_labels.npy'))

if __name__ == '__main__':
    main()
