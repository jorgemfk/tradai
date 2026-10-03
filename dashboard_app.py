"""
Dashboard interactivo (Streamlit) para explorar las oportunidades de compra,
su riesgo, los patrones de velas detectados y el estado emocional de la
población de agentes — con filtros, en vez de una imagen estática.

Uso:
    streamlit run dashboard_app.py

Lee los reportes que genera `python main.py hoy` (output/reporte_*.json),
el historial de la población (state/population_history.json) y la propia
población guardada (state/population.json). No vuelve a descargar datos ni
corre el ciclo diario: solo visualiza lo que ya se calculó.
"""
import glob
import json
import os

import pandas as pd
import streamlit as st

import config
import storage
from population import Population

st.set_page_config(page_title="Agentes evolutivos de trading", layout="wide", page_icon="📈")

RISK_LABELS = {"alto": "🔴 Alto", "bajo": "🟢 Bajo"}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def list_report_files():
    files = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "reporte_*.json")))
    return files


@st.cache_data(ttl=30)
def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def opportunities_dataframe(oportunidades: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(oportunidades)
    if df.empty:
        return df
    if "patrones" not in df.columns:
        df["patrones"] = [[] for _ in range(len(df))]
    df["patrones"] = df["patrones"].apply(lambda p: p if isinstance(p, list) else [])
    return df


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.title("📈 Agentes evolutivos de trading")
    st.caption("Oportunidades de compra, riesgo, patrones de velas y estado emocional de la población — "
               "generado por `main.py hoy`, explorado aquí con filtros.")

    report_files = list_report_files()
    if not report_files:
        st.warning("Todavía no hay reportes guardados. Corre `python main.py hoy` al menos una vez "
                   "(después de `python main.py entrenar`) y luego recarga esta página.")
        return

    fechas = [os.path.basename(f).replace("reporte_", "").replace(".json", "") for f in report_files]

    st.sidebar.header("📅 Reporte")
    fecha_sel = st.sidebar.selectbox("Fecha", fechas[::-1], index=0)
    result = load_report(report_files[fechas.index(fecha_sel)])

    oportunidades = result.get("oportunidades", [])
    stats = result.get("stats_poblacion", {})
    evolucion = result.get("evolucion", {})
    df_op = opportunities_dataframe(oportunidades)

    # --- Filtros ------------------------------------------------------------
    st.sidebar.header("🔎 Filtros")
    if not df_op.empty:
        tickers_disponibles = sorted(df_op["ticker"].unique())
    else:
        tickers_disponibles = []
    tickers_sel = st.sidebar.multiselect("Ticker", tickers_disponibles, default=tickers_disponibles)
    riesgo_sel = st.sidebar.multiselect("Riesgo", ["alto", "bajo"], default=["alto", "bajo"],
                                          format_func=lambda r: RISK_LABELS.get(r, r))
    solo_con_patron = st.sidebar.checkbox("Solo con patrón de velas detectado", value=False)
    confianza_min = st.sidebar.slider("Confianza mínima de los agentes (%)", 0, 100, 0)

    if not df_op.empty:
        df_filt = df_op[df_op["ticker"].isin(tickers_sel) & df_op["riesgo"].isin(riesgo_sel)]
        df_filt = df_filt[df_filt["confianza_promedio"] * 100 >= confianza_min]
        if solo_con_patron:
            df_filt = df_filt[df_filt["patrones"].apply(len) > 0]
    else:
        df_filt = df_op

    # --- KPIs -----------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Oportunidades hoy", len(df_op))
    c2.metric("😀 Agentes felices", stats.get("felices", 0))
    c3.metric("😟 Agentes tristes", stats.get("tristes", 0))
    c4.metric("Felicidad promedio", f"{stats.get('felicidad_promedio', 0):.1f}")
    c5.metric("Generación", stats.get("generacion", 0))

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"Oportunidades de compra — {fecha_sel}")
        if df_filt.empty:
            st.info("No hay oportunidades que cumplan los filtros seleccionados.")
        else:
            show = df_filt.copy()
            show["Patrones"] = show["patrones"].apply(lambda p: ", ".join(p) if p else "—")
            show["Confianza"] = (show["confianza_promedio"] * 100).round(0).astype(int).astype(str) + "%"
            show["Caída desde máx."] = show["caida_desde_maximo_pct"].round(1).astype(str) + "%"
            show["Riesgo"] = show["riesgo"].map(RISK_LABELS).fillna(show["riesgo"])
            show["Precio"] = show["precio"].apply(lambda v: f"${v:,.2f}")
            show = show.rename(columns={"ticker": "Ticker", "n_agentes_compran": "Agentes que compran"})
            show = show.sort_values("confianza_promedio", ascending=False)
            st.dataframe(
                show[["Ticker", "Precio", "Caída desde máx.", "Riesgo", "Patrones",
                      "Agentes que compran", "Confianza"]],
                use_container_width=True, hide_index=True,
            )

            with st.expander("📊 Confianza promedio por ticker"):
                chart_df = show[["Ticker", "confianza_promedio"]].set_index("Ticker") * 100
                st.bar_chart(chart_df)

    with col_right:
        st.subheader("Estado emocional")
        felices, tristes = stats.get("felices", 0), stats.get("tristes", 0)
        if felices + tristes > 0:
            mood_df = pd.DataFrame({"Agentes": [felices, tristes]}, index=["😀 Felices", "😟 Tristes"])
            st.bar_chart(mood_df)
        win_rate = stats.get("win_rate_promedio")
        if win_rate is not None:
            st.metric("Win rate promedio histórico", f"{win_rate * 100:.0f}%")
        st.caption(f"Hoy: murieron {evolucion.get('murieron', 0)} agentes, "
                   f"nacieron {evolucion.get('nacieron', 0)} nuevos "
                   f"(población total: {evolucion.get('poblacion_total', stats.get('total_agentes', 0))}).")

    st.divider()

    # --- Historial de felicidad promedio ------------------------------------
    st.subheader("Evolución histórica de la población")
    history = storage.load_history()
    if history:
        hist_df = pd.DataFrame(history)
        tab1, tab2 = st.tabs(["Felicidad promedio", "Felices vs. tristes"])
        with tab1:
            st.line_chart(hist_df.set_index("fecha")[["felicidad_promedio"]])
        with tab2:
            if {"felices", "tristes"}.issubset(hist_df.columns):
                st.line_chart(hist_df.set_index("fecha")[["felices", "tristes"]])
            else:
                st.info("Este historial todavía no registra felices/tristes por día.")
    else:
        st.info("Aún no hay suficiente historial para graficar la evolución (corre `main.py hoy` varios días).")

    # --- Detalle de agentes --------------------------------------------------
    with st.expander("🧬 Ver población de agentes (detalle individual)"):
        pop = Population.load_or_create()
        agent_rows = [{
            "ID": a.id,
            "Generación": a.generation,
            "Felicidad": round(a.happiness, 1),
            "Estado": "😀 Feliz" if a.is_happy() else "😟 Triste",
            "Edad (ciclos)": a.age_cycles,
            "Operaciones evaluadas": a.trades_evaluated,
            "Aciertos": a.trades_won,
            "Win rate": f"{(a.trades_won / a.trades_evaluated * 100):.0f}%" if a.trades_evaluated else "—",
            "Umbral de decisión": round(a.genome["threshold"], 2),
        } for a in pop.agents]
        if agent_rows:
            df_agents = pd.DataFrame(agent_rows).sort_values("Felicidad", ascending=False)
            st.dataframe(df_agents, use_container_width=True, hide_index=True)
        else:
            st.info("No hay población guardada todavía.")


if __name__ == "__main__":
    main()
