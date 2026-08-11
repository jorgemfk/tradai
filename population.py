"""
Maneja la población completa de agentes: generación de predicciones diarias,
evaluación de predicciones pasadas (sube/baja la hormona de felicidad),
muerte de los agentes más tristes y reproducción de los más felices.
"""
from __future__ import annotations

import json
import os
import numpy as np

import config
from agent import NeuralAgent


class Population:
    def __init__(self, agents=None, generation: int = 0):
        self.agents = agents if agents is not None else [
            NeuralAgent(generation=0) for _ in range(config.POPULATION_SIZE)
        ]
        self.generation = generation

    # ------------------------------------------------------------------
    # Predicciones
    # ------------------------------------------------------------------
    def predict_ticker(self, ticker: str, feature_row: np.ndarray, price: float,
                        risk: str | None, date_str: str) -> list[dict]:
        """Cada agente vota sobre esta fila de features. Devuelve la lista de
        predicciones de COMPRA (solo se registran las positivas)."""
        predictions = []
        for a in self.agents:
            buy, prob = a.decide(feature_row)
            if buy and risk is not None:
                predictions.append({
                    "agent_id": a.id,
                    "ticker": ticker,
                    "date": date_str,
                    "price": price,
                    "confidence": prob,
                    "risk": risk,
                })
        return predictions

    # ------------------------------------------------------------------
    # Evaluación de resultados pasados
    # ------------------------------------------------------------------
    def evaluate(self, pending: list[dict], price_lookup) -> tuple[list[dict], list[dict]]:
        """pending: lista de predicciones guardadas hace EVAL_HORIZON_DAYS.
        price_lookup(ticker, date_str) -> precio actual o None si no disponible.
        Devuelve (evaluadas, siguen_pendientes)."""
        by_id = {a.id: a for a in self.agents}
        evaluated, still_pending = [], []
        for pred in pending:
            current_price = price_lookup(pred["ticker"])
            if current_price is None:
                still_pending.append(pred)
                continue
            pct_change = (current_price - pred["price"]) / pred["price"]
            agent = by_id.get(pred["agent_id"])
            if agent is not None:
                agent.apply_result(pct_change)
            pred_eval = dict(pred)
            pred_eval["result_price"] = current_price
            pred_eval["pct_change"] = pct_change
            pred_eval["outcome"] = "acierto" if pct_change > 0 else "fallo"
            evaluated.append(pred_eval)
        return evaluated, still_pending

    # ------------------------------------------------------------------
    # Evolución: mueren los más tristes, nacen hijos de los más felices
    # ------------------------------------------------------------------
    def evolve(self) -> dict:
        for a in self.agents:
            a.age_cycles += 1

        alive = [a for a in self.agents if not a.is_dead()]
        dead = [a for a in self.agents if a.is_dead()]

        n_missing = config.POPULATION_SIZE - len(alive)
        children = []
        if n_missing > 0 and alive:
            ranked = sorted(alive, key=lambda a: a.happiness, reverse=True)
            top = ranked[: max(2, len(ranked) // 2)]
            self.generation += 1
            for _ in range(n_missing):
                if len(top) >= 2:
                    pa, pb = np.random.default_rng().choice(top, size=2, replace=False)
                    child = NeuralAgent.crossover(pa, pb, self.generation)
                else:
                    child = top[0].mutate(self.generation)
                children.append(child)
        elif n_missing > 0 and not alive:
            # extinción total: repoblar desde cero
            children = [NeuralAgent(generation=self.generation + 1)
                        for _ in range(config.POPULATION_SIZE)]
            self.generation += 1

        self.agents = alive + children
        return {
            "murieron": len(dead),
            "nacieron": len(children),
            "poblacion_total": len(self.agents),
        }

    # ------------------------------------------------------------------
    # Estadísticas para el dashboard
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        happy = [a for a in self.agents if a.is_happy()]
        sad = [a for a in self.agents if not a.is_happy()]
        avg_happiness = float(np.mean([a.happiness for a in self.agents])) if self.agents else 0.0
        win_rates = [a.trades_won / a.trades_evaluated for a in self.agents if a.trades_evaluated > 0]
        return {
            "total_agentes": len(self.agents),
            "felices": len(happy),
            "tristes": len(sad),
            "felicidad_promedio": avg_happiness,
            "win_rate_promedio": float(np.mean(win_rates)) if win_rates else None,
            "generacion": self.generation,
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def save(self, path: str = config.POPULATION_FILE):
        data = {
            "generation": self.generation,
            "agents": [a.to_dict() for a in self.agents],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_or_create(cls, path: str = config.POPULATION_FILE) -> "Population":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        agents = [NeuralAgent.from_dict(d) for d in data["agents"]]
        return cls(agents=agents, generation=data.get("generation", 0))
