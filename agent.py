"""
Agente neuronal individual. Cada agente es una red neuronal pequeña
(MLP de una capa oculta) escrita a mano con numpy, con:
  - un "genoma" (los pesos) que se puede mutar / cruzar,
  - una hormona de "felicidad" que sube o baja según el resultado real
    de sus decisiones de compra,
  - un umbral de decisión propio (también forma parte del genoma).
"""
from __future__ import annotations

import itertools
import numpy as np

import config

_id_counter = itertools.count(1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NeuralAgent:
    def __init__(self, genome: dict | None = None, agent_id: int | None = None,
                 generation: int = 0, happiness: float = config.INITIAL_HAPPINESS):
        self.id = agent_id if agent_id is not None else next(_id_counter)
        self.generation = generation
        self.happiness = happiness
        self.age_cycles = 0
        self.trades_evaluated = 0
        self.trades_won = 0
        self.genome = genome if genome is not None else self._random_genome()

    # ------------------------------------------------------------------
    # Genoma / red neuronal
    # ------------------------------------------------------------------
    @staticmethod
    def _random_genome() -> dict:
        n_in, n_hid = config.N_FEATURES, config.HIDDEN_SIZE
        rng = np.random.default_rng()
        return {
            "w1": rng.normal(0, 0.6, size=(n_in, n_hid)),
            "b1": rng.normal(0, 0.1, size=(n_hid,)),
            "w2": rng.normal(0, 0.6, size=(n_hid,)),
            "b2": rng.normal(0, 0.1),
            # umbral de decisión propio del agente (personalidad: más o menos arriesgado)
            "threshold": float(np.clip(rng.normal(config.BUY_PROB_THRESHOLD_BASE, 0.07), 0.35, 0.85)),
        }

    def forward(self, features: np.ndarray) -> float:
        """features: vector 1D de tamaño N_FEATURES -> probabilidad de compra [0,1]."""
        g = self.genome
        hidden = np.tanh(features @ g["w1"] + g["b1"])
        out = _sigmoid(hidden @ g["w2"] + g["b2"])
        return float(out)

    def decide(self, features: np.ndarray) -> tuple[bool, float]:
        prob = self.forward(features)
        return prob >= self.genome["threshold"], prob

    # ------------------------------------------------------------------
    # Hormona de felicidad
    # ------------------------------------------------------------------
    def apply_result(self, pct_change: float):
        """Actualiza la felicidad del agente según el resultado real de una
        compra que sugirió `config.EVAL_HORIZON_DAYS` días atrás.
        pct_change > 0  -> el mercado subió tras su sugerencia (acierto)
        pct_change < 0  -> el mercado siguió cayendo (fallo)
        """
        delta = pct_change * config.GAIN_SCALE
        self.happiness = float(np.clip(self.happiness + delta, 0, config.MAX_HAPPINESS))
        self.trades_evaluated += 1
        if pct_change > 0:
            self.trades_won += 1

    def is_happy(self) -> bool:
        return self.happiness >= config.INITIAL_HAPPINESS

    def is_dead(self) -> bool:
        return self.happiness <= config.DEATH_HAPPINESS

    # ------------------------------------------------------------------
    # Evolución: mutación y cruce
    # ------------------------------------------------------------------
    def mutate(self, generation: int) -> "NeuralAgent":
        rng = np.random.default_rng()
        new_genome = {}
        for k, v in self.genome.items():
            arr = np.array(v, dtype=float)
            mask = rng.random(arr.shape) < config.MUTATION_RATE
            noise = rng.normal(0, config.MUTATION_STRENGTH, size=arr.shape)
            new_genome[k] = arr + mask * noise
        new_genome["threshold"] = float(np.clip(new_genome["threshold"], 0.3, 0.9))
        child = NeuralAgent(genome=new_genome, generation=generation)
        return child

    @staticmethod
    def crossover(parent_a: "NeuralAgent", parent_b: "NeuralAgent", generation: int) -> "NeuralAgent":
        rng = np.random.default_rng()
        new_genome = {}
        for k in parent_a.genome:
            a, b = np.array(parent_a.genome[k], dtype=float), np.array(parent_b.genome[k], dtype=float)
            mask = rng.random(a.shape) < 0.5
            new_genome[k] = np.where(mask, a, b)
        child = NeuralAgent(genome=new_genome, generation=generation)
        # aplica una mutación leve extra para mantener diversidad
        return child.mutate(generation)

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "happiness": self.happiness,
            "age_cycles": self.age_cycles,
            "trades_evaluated": self.trades_evaluated,
            "trades_won": self.trades_won,
            "genome": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in self.genome.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuralAgent":
        genome = {k: (np.array(v) if k != "threshold" else v) for k, v in d["genome"].items()}
        agent = cls(genome=genome, agent_id=d["id"], generation=d["generation"],
                     happiness=d["happiness"])
        agent.age_cycles = d.get("age_cycles", 0)
        agent.trades_evaluated = d.get("trades_evaluated", 0)
        agent.trades_won = d.get("trades_won", 0)
        return agent
