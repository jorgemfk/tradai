"""
Orquesta dos modos de ejecución:

  train_historical()  -> recorre varios años de histórico día a día,
                          dejando que la población evolucione (aprenda,
                          muera, se reproduzca) antes de usarse "en vivo".

  run_daily_cycle()    -> ejecución de un solo día real: genera las
                          oportunidades de hoy, evalúa las predicciones de
                          hace `EVAL_HORIZON_DAYS` días y evoluciona la población.

Ambos modos comparten la misma lógica de features / riesgo / agentes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import storage
from features import build_feature_table, classify_opportunity, FEATURE_COLUMNS
from population import Population


def _align_tables(feature_tables: dict) -> dict:
    """Reindexa todas las tablas de features a un índice de fechas común
    (unión de todas las fechas disponibles), rellenando huecos hacia adelante
    (por ejemplo cripto opera fines de semana y acciones no)."""
    common_index = sorted(set().union(*[df.index for df in feature_tables.values()]))
    aligned = {}
    for ticker, df in feature_tables.items():
        aligned[ticker] = df.reindex(common_index).ffill()
    return aligned, pd.DatetimeIndex(common_index)


def build_all_features(raw_data: dict) -> dict:
    return {t: build_feature_table(df) for t, df in raw_data.items()}


# ---------------------------------------------------------------------------
# Entrenamiento histórico (evolución sobre varios años)
# ---------------------------------------------------------------------------
def train_historical(raw_data: dict, population: Population | None = None,
                      verbose: bool = True) -> Population:
    feature_tables = build_all_features(raw_data)
    aligned, common_index = _align_tables(feature_tables)
    population = population or Population()

    pending = []  # predicciones esperando evaluación (en memoria durante el entrenamiento)
    start = config.MIN_WARMUP_DAYS
    total_opportunities = 0

    for i in range(start, len(common_index)):
        date = common_index[i]
        date_str = str(date.date())

        # 1) generar oportunidades / predicciones del día
        for ticker, df in aligned.items():
            row = df.loc[date]
            if row.isna().any():
                continue
            is_opp, risk = classify_opportunity(row)
            if not is_opp:
                continue
            feat_vec = row[FEATURE_COLUMNS].values.astype(float)
            preds = population.predict_ticker(ticker, feat_vec, float(row["close"]), risk, date_str)
            for p in preds:
                p["eval_index"] = i + config.EVAL_HORIZON_DAYS
            pending.extend(preds)
            total_opportunities += len(preds)

        # 2) evaluar lo que ya cumplió el horizonte de evaluación
        due = [p for p in pending if p["eval_index"] <= i]
        if due:
            pending = [p for p in pending if p["eval_index"] > i]

            def price_lookup(ticker, _i=i):
                try:
                    val = aligned[ticker]["close"].iloc[_i]
                    return float(val) if not pd.isna(val) else None
                except Exception:  # noqa: BLE001
                    return None

            population.evaluate(due, price_lookup)

        # 3) evolucionar la población periódicamente
        if (i - start) % config.EVAL_HORIZON_DAYS == 0 and i > start:
            result = population.evolve()
            if verbose and (i - start) % (config.EVAL_HORIZON_DAYS * 20) == 0:
                stats = population.stats()
                print(f"[train] {date_str} gen={stats['generacion']} "
                      f"felices={stats['felices']} tristes={stats['tristes']} "
                      f"felicidad_prom={stats['felicidad_promedio']:.1f} "
                      f"murieron={result['murieron']}")

    if verbose:
        print(f"[train] Entrenamiento histórico terminado. "
              f"{total_opportunities} oportunidades procesadas en {len(common_index) - start} días.")
    return population


# ---------------------------------------------------------------------------
# Ciclo diario en vivo
# ---------------------------------------------------------------------------
def _trading_days_elapsed(df: pd.DataFrame, from_date_str: str) -> int:
    try:
        loc = df.index.get_indexer([pd.Timestamp(from_date_str)], method="ffill")[0]
    except Exception:  # noqa: BLE001
        return 0
    if loc < 0:
        return 0
    return (len(df) - 1) - loc


def run_daily_cycle(raw_data: dict, population: Population) -> dict:
    feature_tables = build_all_features(raw_data)
    pending = storage.load_pending()

    today_str = None
    todays_opportunities = []

    # 1) predicciones de hoy (última fila disponible de cada ticker)
    for ticker, df in feature_tables.items():
        if df.empty:
            continue
        row = df.iloc[-1]
        date_str = str(df.index[-1].date())
        today_str = date_str
        is_opp, risk = classify_opportunity(row)
        if not is_opp:
            continue
        feat_vec = row[FEATURE_COLUMNS].values.astype(float)
        preds = population.predict_ticker(ticker, feat_vec, float(row["close"]), risk, date_str)
        pending.extend(preds)
        if preds:
            confidences = [p["confidence"] for p in preds]
            todays_opportunities.append({
                "ticker": ticker,
                "riesgo": risk,
                "precio": float(row["close"]),
                "caida_desde_maximo_pct": float(row["drop_from_high"] * 100),
                "n_agentes_compran": len(preds),
                "confianza_promedio": float(np.mean(confidences)),
            })

    # 2) evaluar predicciones que ya cumplieron el horizonte
    due, still_pending = [], []
    for p in pending:
        df = feature_tables.get(p["ticker"])
        if df is None or df.empty:
            still_pending.append(p)
            continue
        elapsed = _trading_days_elapsed(df, p["date"])
        if elapsed >= config.EVAL_HORIZON_DAYS:
            due.append(p)
        else:
            still_pending.append(p)

    def price_lookup(ticker):
        df = feature_tables.get(ticker)
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])

    evaluated, still_pending_2 = population.evaluate(due, price_lookup)
    still_pending.extend(still_pending_2)
    storage.save_pending(still_pending)

    # 3) evolución de la población (mueren los más tristes, nacen nuevos)
    evolve_result = population.evolve()

    stats = population.stats()
    todays_opportunities.sort(key=lambda o: o["confianza_promedio"], reverse=True)

    result = {
        "fecha": today_str,
        "oportunidades": todays_opportunities,
        "evaluadas_hoy": evaluated,
        "evolucion": evolve_result,
        "stats_poblacion": stats,
    }
    storage.append_history({"fecha": today_str, **stats, **evolve_result})
    return result
