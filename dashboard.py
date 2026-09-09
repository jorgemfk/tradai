"""
Genera el reporte visual diario:
  - tabla de oportunidades de compra con su riesgo (alto/bajo)
  - estado emocional de la población (agentes felices vs tristes)
  - evolución histórica de la felicidad promedio (si hay historial)
Guarda un PNG en output/ con la fecha, y también un JSON con los datos crudos.
"""
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import storage

RISK_COLORS = {"alto": "#e05555", "bajo": "#3aa65b"}


def render_daily_report(result: dict, save: bool = True) -> str:
    fecha = result["fecha"] or datetime.now().strftime("%Y-%m-%d")
    oportunidades = result["oportunidades"]
    stats = result["stats_poblacion"]
    history = storage.load_history()

    fig = plt.figure(figsize=(13, 8.5), facecolor="#0f1116")
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], hspace=0.35, wspace=0.28)

    # --- Panel 1: tabla de oportunidades -----------------------------------
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.set_facecolor("#0f1116")
    ax_table.axis("off")
    ax_table.set_title(f"Oportunidades de compra detectadas — {fecha}",
                        color="white", fontsize=15, loc="left", fontweight="bold")

    if oportunidades:
        rows = oportunidades[:18]
        col_labels = ["Ticker", "Precio", "Caída desde máx.", "Riesgo", "Martillo",
                      "Agentes que compran", "Confianza"]
        table_data = []
        for o in rows:
            table_data.append([
                o["ticker"],
                f"${o['precio']:.2f}",
                f"{o['caida_desde_maximo_pct']:.1f}%",
                o["riesgo"].upper(),
                "Sí" if o.get("patron_martillo") else "No",
                str(o["n_agentes_compran"]),
                f"{o['confianza_promedio']*100:.0f}%",
            ])
        tbl = ax_table.table(cellText=table_data, colLabels=col_labels,
                              loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10.5)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#333844")
            if r == 0:
                cell.set_facecolor("#1c2029")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#171a21")
                cell.set_text_props(color="white")
                if c == 3:
                    riesgo = table_data[r - 1][3].lower()
                    cell.set_text_props(color=RISK_COLORS.get(riesgo, "white"), fontweight="bold")
                if c == 4 and table_data[r - 1][4] == "Sí":
                    cell.set_text_props(color="#f2b84b", fontweight="bold")
    else:
        ax_table.text(0.02, 0.6, "No se detectaron oportunidades de compra hoy.",
                       color="#aaaaaa", fontsize=12)

    # --- Panel 2: felices vs tristes (dona) ---------------------------------
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_pie.set_facecolor("#0f1116")
    felices, tristes = stats["felices"], stats["tristes"]
    if felices + tristes > 0:
        ax_pie.pie([felices, tristes], labels=[f"Felices ({felices})", f"Tristes ({tristes})"],
                    colors=["#3aa65b", "#e05555"], autopct="%1.0f%%",
                    textprops={"color": "white"}, wedgeprops={"width": 0.45})
    ax_pie.set_title("Estado emocional de los agentes", color="white", fontsize=12, fontweight="bold")

    # --- Panel 3: evolución de felicidad promedio en el tiempo -------------
    ax_line = fig.add_subplot(gs[1, 1])
    ax_line.set_facecolor("#0f1116")
    if history:
        fechas = [h["fecha"] for h in history][-60:]
        felicidad = [h["felicidad_promedio"] for h in history][-60:]
        ax_line.plot(range(len(felicidad)), felicidad, color="#5aa9ff", linewidth=2)
        ax_line.fill_between(range(len(felicidad)), felicidad, color="#5aa9ff", alpha=0.15)
        step = max(1, len(fechas) // 6)
        ax_line.set_xticks(range(0, len(fechas), step))
        ax_line.set_xticklabels([fechas[i] for i in range(0, len(fechas), step)],
                                  color="#aaaaaa", fontsize=8, rotation=30, ha="right")
    ax_line.set_ylim(0, config.MAX_HAPPINESS)
    ax_line.set_title("Felicidad promedio de la población (histórico)",
                        color="white", fontsize=12, fontweight="bold")
    ax_line.tick_params(colors="#aaaaaa")
    for spine in ax_line.spines.values():
        spine.set_color("#333844")

    fig.suptitle("Agentes evolutivos de trading — reporte diario",
                  color="white", fontsize=17, fontweight="bold", y=0.985)

    out_path = os.path.join(config.OUTPUT_DIR, f"reporte_{fecha}.png")
    if save:
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        with open(os.path.join(config.OUTPUT_DIR, f"reporte_{fecha}.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    plt.close(fig)
    return out_path
