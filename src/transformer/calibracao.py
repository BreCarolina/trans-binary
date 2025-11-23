#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibracao.py

Calibradores de probabilidade para o projeto trans-binary.

Suporta:
 - Platt Scaling (regressão logística sobre logits)
 - Isotonic Regression (não-paramétrico sobre probabilidades)
 - Temperature Scaling (ajuste do softmax / logits via T)

Classe principal:
 - CalibradorProbabilidades:
     fit(logits_or_probs, labels, method='platt'|'isotonic'|'temperature'|'all')
     predict(logits_or_probs)
     save(path_dir)
     load(path_dir)

Uso rápido:
    python3 -c "from src.transformer.calibracao import CalibradorProbabilidades; print('OK')"

Observações:
 - Platt e Temperature assumem entrada em *logits* (ou scores não normalizados).
 - Isotonic assume entrada em probabilidades no intervalo [0,1].
 - Salvamento: platt -> JSON com coeficientes; isotonic -> joblib pickle; temperature -> JSON com T.
"""

from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
import joblib

# -------------------------
# Helpers
# -------------------------
def _softmax_1d(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax para vetor de logits binários/2-classes.
    Se logits forem shape (N, ) assume-se que representam logit da classe 1.
    Retorna probabilidades da classe 1.
    """
    l = np.asarray(logits)
    if l.ndim == 2 and l.shape[1] == 2:
        # logits 2-d: use softmax per linha and return prob class 1
        z = l / float(temperature)
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        probs = exp[:, 1] / (exp.sum(axis=1))
        return probs
    else:
        # treat as single logit -> sigmoid(logit / T)
        return 1.0 / (1.0 + np.exp(-(l / float(temperature))))

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# -------------------------
# Calibrators implementations
# -------------------------
class PlattCalibrator:
    """Platt scaling using a sklearn LogisticRegression (solver liblinear)."""
    def __init__(self):
        self.model: Optional[LogisticRegression] = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """
        Treina regressão logística sobre *logits*.
        logits: shape (N,) or (N,2). Se (N,2) usa o logit relativo à classe 1 (col 1) minus col 0.
        labels: {0,1}
        """
        logits = np.asarray(logits)
        labels = np.asarray(labels).ravel()
        if logits.ndim == 2 and logits.shape[1] == 2:
            X = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
        else:
            X = logits.reshape(-1, 1)
        # use liblinear for small logistic fit
        self.model = LogisticRegression(solver='liblinear', penalty='l2', C=1.0, max_iter=1000)
        self.model.fit(X, labels)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PlattCalibrator not fitted")
        logits = np.asarray(logits)
        if logits.ndim == 2 and logits.shape[1] == 2:
            X = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
        else:
            X = logits.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        _ensure_dir(os.path.dirname(path) or '.')
        if self.model is None:
            raise RuntimeError("PlattCalibrator not fitted")
        # save coef and intercept for reproducibility
        payload = {
            'coef': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist(),
            'classes': self.model.classes_.tolist()
        }
        with open(path, 'w') as f:
            json.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, 'r') as f:
            payload = json.load(f)
        # rehydrate a sklearn LogisticRegression with the saved params
        model = LogisticRegression(solver='liblinear')
        model.coef_ = np.asarray(payload['coef'])
        model.intercept_ = np.asarray(payload['intercept'])
        model.classes_ = np.asarray(payload['classes'])
        # sklearn requires `n_features_in_` attribute for some versions
        model.n_features_in_ = model.coef_.shape[1]
        self.model = model

class IsotonicCalibrator:
    """Isotonic regression calibrator. Works on probabilities."""
    def __init__(self):
        self.model: Optional[IsotonicRegression] = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        probs = np.asarray(probs).ravel()
        labels = np.asarray(labels).ravel()
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.model.fit(probs, labels)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator not fitted")
        probs = np.asarray(probs).ravel()
        return self.model.predict(probs)

    def save(self, path: str) -> None:
        _ensure_dir(os.path.dirname(path) or '.')
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator not fitted")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)

class TemperatureCalibrator:
    """Temperature scaling: busca por T que minimiza NLL no conjunto de validação."""
    def __init__(self):
        self.temperature: Optional[float] = None

    def fit(self, logits: np.ndarray, labels: np.ndarray, search_grid: Optional[np.ndarray] = None) -> None:
        """
        Encontra temperatura T por search grid (robusto, sem dependência em scipy).
        logits: shape (N,) ou (N,2)
        labels: 0/1
        """
        logits = np.asarray(logits)
        labels = np.asarray(labels).ravel()
        if search_grid is None:
            # grid log-spaced entre 0.05 e 5.0, mais densidade próxima de 1.0
            search_grid = np.concatenate([
                np.linspace(0.05, 0.5, 10),
                np.linspace(0.6, 1.5, 10),
                np.linspace(1.6, 5.0, 12)
            ])
        best_T = 1.0
        best_loss = float('inf')
        for T in search_grid:
            probs = _softmax_1d(logits, temperature=T)
            # numeric safety for log_loss
            try:
                loss = log_loss(labels, probs, labels=[0,1])
            except ValueError:
                # bad probs maybe all 0 or 1; add epsilon
                eps = 1e-12
                probs = np.clip(probs, eps, 1 - eps)
                loss = log_loss(labels, probs, labels=[0,1])
            if loss < best_loss:
                best_loss = loss
                best_T = float(T)
        self.temperature = float(best_T)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("TemperatureCalibrator not fitted")
        return _softmax_1d(logits, temperature=self.temperature)

    def save(self, path: str) -> None:
        _ensure_dir(os.path.dirname(path) or '.')
        if self.temperature is None:
            raise RuntimeError("TemperatureCalibrator not fitted")
        with open(path, 'w') as f:
            json.dump({'temperature': float(self.temperature)}, f)

    def load(self, path: str) -> None:
        with open(path, 'r') as f:
            payload = json.load(f)
        self.temperature = float(payload['temperature'])

# -------------------------
# Unified API
# -------------------------
class CalibradorProbabilidades:
    """
    API unificada para calibração.
    method: 'platt', 'isotonic', 'temperature', or 'all' to fit all calibrators and keep them.
    """
    def __init__(self):
        self.platt = PlattCalibrator()
        self.isotonic = IsotonicCalibrator()
        self.temperature = TemperatureCalibrator()
        self.fitted: Dict[str, bool] = {'platt': False, 'isotonic': False, 'temperature': False}

    def fit(self, logits_or_probs: np.ndarray, labels: np.ndarray, method: str = 'all') -> None:
        """
        Fit calibrator(s).
        - For 'platt' and 'temperature' pass logits_or_probs as logits (shape (N,) or (N,2))
        - For 'isotonic' pass logits_or_probs as probabilities in [0,1]
        - method can be 'platt', 'isotonic', 'temperature', or 'all'
        """
        method = method.lower()
        if method not in {'platt', 'isotonic', 'temperature', 'all'}:
            raise ValueError("method must be 'platt'|'isotonic'|'temperature'|'all'")

        if method in {'platt', 'all'}:
            self.platt.fit(logits_or_probs, labels)
            self.fitted['platt'] = True

        if method in {'temperature', 'all'}:
            self.temperature.fit(logits_or_probs, labels)
            self.fitted['temperature'] = True

        if method in {'isotonic', 'all'}:
            # If user passed logits and wants isotonic, convert via sigmoid
            arr = np.asarray(logits_or_probs)
            if arr.ndim == 2 and arr.shape[1] == 2:
                probs = _softmax_1d(arr, temperature=1.0)
            elif arr.ndim == 1 and (arr.max() > 1 or arr.min() < 0):
                # values look like logits -> convert via sigmoid
                probs = 1.0 / (1.0 + np.exp(-arr))
            else:
                probs = arr
            self.isotonic.fit(probs, labels)
            self.fitted['isotonic'] = True

    def predict(self, logits_or_probs: np.ndarray, method: str = 'platt') -> np.ndarray:
        """
        Returns calibrated probabilities according to method.
        - method: 'platt'|'isotonic'|'temperature'
        - Input expected:
            platt/temperature: logits (N,) or (N,2)
            isotonic: probabilities (N,)
        """
        method = method.lower()
        if method == 'platt':
            if not self.fitted['platt']:
                raise RuntimeError("Platt calibrator not fitted")
            return self.platt.predict(logits_or_probs)
        elif method == 'temperature':
            if not self.fitted['temperature']:
                raise RuntimeError("Temperature calibrator not fitted")
            return self.temperature.predict(logits_or_probs)
        elif method == 'isotonic':
            if not self.fitted['isotonic']:
                raise RuntimeError("Isotonic calibrator not fitted")
            # If input are logits convert to probs first
            arr = np.asarray(logits_or_probs)
            if arr.ndim == 2 and arr.shape[1] == 2:
                probs = _softmax_1d(arr, temperature=1.0)
            elif arr.ndim == 1 and (arr.max() > 1 or arr.min() < 0):
                probs = 1.0 / (1.0 + np.exp(-arr))
            else:
                probs = arr
            return self.isotonic.predict(probs)
        else:
            raise ValueError("method must be 'platt'|'isotonic'|'temperature'")

    def save(self, dir_path: str) -> None:
        """
        Salva os calibradores existentes em dir_path.
        Arquivos gerados:
         - platt.json
         - isotonic.joblib
         - temperature.json
        """
        _ensure_dir(dir_path)
        if self.fitted.get('platt', False):
            platt_path = os.path.join(dir_path, 'platt.json')
            self.platt.save(platt_path)
        if self.fitted.get('isotonic', False):
            isotonic_path = os.path.join(dir_path, 'isotonic.joblib')
            self.isotonic.save(isotonic_path)
        if self.fitted.get('temperature', False):
            temp_path = os.path.join(dir_path, 'temperature.json')
            self.temperature.save(temp_path)

    def load(self, dir_path: str) -> None:
        """Carrega calibradores presentes em dir_path (se existir)."""
        # platt
        platt_path = os.path.join(dir_path, 'platt.json')
        if os.path.exists(platt_path):
            self.platt.load(platt_path)
            self.fitted['platt'] = True
        isotonic_path = os.path.join(dir_path, 'isotonic.joblib')
        if os.path.exists(isotonic_path):
            self.isotonic.load(isotonic_path)
            self.fitted['isotonic'] = True
        temp_path = os.path.join(dir_path, 'temperature.json')
        if os.path.exists(temp_path):
            self.temperature.load(temp_path)
            self.fitted['temperature'] = True

# -------------------------
# CLI (opcional)
# -------------------------
def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(description='Calibrador simples (fit e salva artefatos).')
    p.add_argument('--logits', required=True, help='.npy com logits (N,) ou (N,2)')
    p.add_argument('--labels', required=True, help='.npy com labels binários (0/1)')
    p.add_argument('--out', required=False, default='models/transformer/calibration', help='diretório de saída')
    p.add_argument('--methods', required=False, default='all', help="vírgula-separado: platt,isotonic,temperature ou all")
    return p.parse_args()

def _cli_main():
    args = _parse_cli()
    logits = np.load(args.logits)
    labels = np.load(args.labels)
    methods = [m.strip().lower() for m in args.methods.split(',')]
    if 'all' in methods:
        methods = ['all']
    c = CalibradorProbabilidades()
    c.fit(logits, labels, method='all' if methods == ['all'] else methods[0])
    c.save(args.out)
    print('Calibradores salvos em:', args.out)

if __name__ == '__main__':
    _cli_main()

