"""
Persistencia simple en JSON (sin base de datos) para:
  - predicciones pendientes de evaluar (compradas hace < EVAL_HORIZON_DAYS)
  - historial de estadísticas de la población (para graficar evolución)
"""
import json
import os

import config


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_pending() -> list[dict]:
    return load_json(config.PENDING_FILE, [])


def save_pending(pending: list[dict]):
    save_json(config.PENDING_FILE, pending)


def append_history(entry: dict):
    hist = load_json(config.HISTORY_LOG_FILE, [])
    hist.append(entry)
    save_json(config.HISTORY_LOG_FILE, hist)


def load_history() -> list[dict]:
    return load_json(config.HISTORY_LOG_FILE, [])
