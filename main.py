"""
Punto de entrada.

Uso:
  python main.py entrenar          -> descarga histórico y entrena/evoluciona
                                        la población desde cero (o continúa si
                                        ya existe una población guardada)
  python main.py hoy                 -> ejecuta el ciclo de un día real:
                                        detecta oportunidades, evalúa
                                        predicciones pasadas, evoluciona la
                                        población y genera el reporte visual.
"""
import sys

import config
import data_provider
from population import Population
from simulator import train_historical, run_daily_cycle
from dashboard import render_daily_report


def cmd_entrenar():
    print(f"Descargando {config.YEARS_HISTORY} años de histórico para "
          f"{len(config.ALL_TICKERS)} activos...")
    raw_data = data_provider.load_all(refresh=True)
    population = Population.load_or_create()
    population = train_historical(raw_data, population=population)
    population.save()
    print(f"Población guardada en {config.POPULATION_FILE}")
    stats = population.stats()
    print(stats)


def cmd_hoy():
    raw_data = data_provider.load_all(refresh=True)
    population = Population.load_or_create()
    if len(population.agents) == 0:
        print("No hay población entrenada. Ejecuta primero: python main.py entrenar")
        return
    result = run_daily_cycle(raw_data, population)
    population.save()

    print(f"\n=== Reporte del {result['fecha']} ===")
    print(f"Oportunidades detectadas: {len(result['oportunidades'])}")
    for o in result["oportunidades"][:10]:
        print(f"  {o['ticker']:8s} riesgo={o['riesgo']:5s} "
              f"precio=${o['precio']:.2f} caida={o['caida_desde_maximo_pct']:.1f}% "
              f"agentes={o['n_agentes_compran']} confianza={o['confianza_promedio']*100:.0f}%")
    print(f"Población: {result['stats_poblacion']}")
    print(f"Evolución: {result['evolucion']}")

    path = render_daily_report(result)
    print(f"\nReporte visual guardado en: {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "hoy"
    if cmd == "entrenar":
        cmd_entrenar()
    elif cmd == "hoy":
        cmd_hoy()
    else:
        print(__doc__)
