"""
Construye el vector de features diario para cada activo y clasifica el
riesgo de una posible oportunidad de compra usando reglas simples
(caída desde máximo reciente + tendencia).
"""
import numpy as np
import pandas as pd

import config


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _prior_downtrend(close: pd.Series, pattern_span: int, trend_lookback: int = 5) -> pd.Series:
    """True si, antes de que empezara el patrón (que ocupa `pattern_span`
    días terminando hoy), el precio venía cayendo. Se usa como requisito de
    contexto: estos patrones solo cuentan como señal de REVERSIÓN si aparecen
    después de una caída."""
    return close.shift(pattern_span) < close.shift(pattern_span + trend_lookback)


def detect_bullish_hammer(df: pd.DataFrame, trend_lookback: int = 5) -> pd.Series:
    """Detecta el patrón de velas 'martillo alcista' (hammer):

      - cuerpo pequeño (apertura/cierre cercanos) ubicado en la parte
        superior del rango del día
      - sombra inferior larga (al menos 2x el tamaño del cuerpo) que
        muestra que el precio fue rechazado tras caer durante la sesión
      - sombra superior corta o inexistente
      - aparece después de una caída reciente (si no, no es un patrón de
        reversión válido, es solo una vela con mecha larga)

    Devuelve una Serie booleana alineada al índice de `df`.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    body = (c - o).abs()
    rng = (h - l)
    rng_safe = rng.replace(0, np.nan)

    upper_body = pd.concat([o, c], axis=1).max(axis=1)
    lower_body = pd.concat([o, c], axis=1).min(axis=1)
    lower_shadow = lower_body - l
    upper_shadow = h - upper_body

    # umbral mínimo de "cuerpo" para evitar división por cero en velas doji
    body_floor = (rng_safe * 0.02).fillna(0)
    body_eff = body.where(body > body_floor, body_floor)

    body_pequeno = (body / rng_safe) <= 0.35
    sombra_inferior_larga = lower_shadow >= (2.0 * body_eff)
    # la sombra superior debe ser corta, tolerando un poco más cuando el
    # cuerpo es muy pequeño (para no descartar martillos "perfectos")
    sombra_superior_corta = upper_shadow <= np.maximum(0.5 * body_eff, 0.12 * rng_safe)

    caida_previa = _prior_downtrend(c, pattern_span=1, trend_lookback=trend_lookback)

    hammer = body_pequeno & sombra_inferior_larga & sombra_superior_corta & caida_previa
    return hammer.fillna(False)


def detect_bullish_engulfing(df: pd.DataFrame, trend_lookback: int = 5) -> pd.Series:
    """Patrón 'envolvente alcista' (bullish engulfing), 2 velas:

      - ayer: vela bajista (cierre < apertura)
      - hoy: vela alcista (cierre > apertura) cuyo cuerpo "envuelve" por
        completo el cuerpo de ayer (abre más abajo que el cierre de ayer y
        cierra más arriba que la apertura de ayer)
      - el cuerpo de hoy es mayor al de ayer (convicción compradora real)
      - aparece después de una caída reciente
    """
    o, c = df["open"], df["close"]

    bajista_ayer = c.shift(1) < o.shift(1)
    alcista_hoy = c > o
    envuelve = (o <= c.shift(1)) & (c >= o.shift(1))
    cuerpo_mayor = (c - o).abs() > (c.shift(1) - o.shift(1)).abs()
    caida_previa = _prior_downtrend(c, pattern_span=2, trend_lookback=trend_lookback)

    engulfing = bajista_ayer & alcista_hoy & envuelve & cuerpo_mayor & caida_previa
    return engulfing.fillna(False)


def detect_morning_star(df: pd.DataFrame, trend_lookback: int = 5) -> pd.Series:
    """Patrón 'estrella de la mañana' (morning star), 3 velas:

      - hace 2 días: vela bajista fuerte (cuerpo grande respecto a su rango)
      - ayer: vela de cuerpo pequeño ("la estrella", indecisión)
      - hoy: vela alcista que cierra por encima de la mitad del cuerpo de
        hace 2 días (recupera buena parte de la caída)
      - aparece después de una caída reciente (antes de la vela bajista fuerte)
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    body1 = o.shift(2) - c.shift(2)                  # positivo si fue bajista
    range1 = (h.shift(2) - l.shift(2)).replace(0, np.nan)
    dia1_bajista_fuerte = (body1 > 0) & ((body1 / range1) >= 0.5)

    body2 = (c.shift(1) - o.shift(1)).abs()
    range2 = (h.shift(1) - l.shift(1)).replace(0, np.nan)
    dia2_cuerpo_pequeno = (body2 / range2) <= 0.3

    punto_medio_dia1 = (o.shift(2) + c.shift(2)) / 2
    dia3_recupera = (c > o) & (c > punto_medio_dia1)

    caida_previa = _prior_downtrend(c, pattern_span=3, trend_lookback=trend_lookback)

    morning_star = dia1_bajista_fuerte & dia2_cuerpo_pequeno & dia3_recupera & caida_previa
    return morning_star.fillna(False)


def detect_indecision_doji(df: pd.DataFrame, trend_lookback: int = 5) -> pd.Series:
    """Doji de indecisión: apertura y cierre casi idénticos (cuerpo
    minúsculo respecto al rango del día), que aparece tras una caída —
    señal de que los vendedores están perdiendo fuerza, aunque todavía sin
    confirmar la reversión."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    cuerpo_minusculo = (body / rng) <= 0.1

    caida_previa = _prior_downtrend(c, pattern_span=1, trend_lookback=trend_lookback)

    doji = cuerpo_minusculo & caida_previa
    return doji.fillna(False)


# Columnas de patrón -> etiqueta legible para tablas y el dashboard
PATTERN_LABELS = {
    "hammer_signal": "Martillo",
    "engulfing_signal": "Envolvente alcista",
    "morning_star_signal": "Estrella de la mañana",
    "doji_signal": "Doji de indecisión",
}
PATTERN_COLUMNS = list(PATTERN_LABELS.keys())


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """A partir del OHLCV diario, arma una tabla de features por día.

    Columnas de salida (en este orden -> N_FEATURES):
      0 drop_from_high   : caída % desde el máximo de 52 semanas (positivo = caída)
      1 sma20_dist        : distancia % del precio a su media de 20 días
      2 sma50_slope       : pendiente % de la media de 50 días (tendencia de fondo)
      3 rsi_norm            : RSI normalizado a [-1, 1]
      4 volatility          : volatilidad (std retornos 20d)
      5 momentum_10d        : retorno acumulado de los últimos 10 días
      6 volume_zscore       : z-score del volumen respecto a su media 20d
      7 hammer_signal       : 1.0 si hoy se formó un martillo alcista
      8 engulfing_signal    : 1.0 si hoy se formó una envolvente alcista
      9 morning_star_signal : 1.0 si hoy se completó una estrella de la mañana
      10 doji_signal          : 1.0 si hoy se formó un doji de indecisión
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    rolling_high = close.rolling(252, min_periods=30).max()
    out["drop_from_high"] = (rolling_high - close) / rolling_high

    sma20 = close.rolling(20, min_periods=10).mean()
    out["sma20_dist"] = (close - sma20) / sma20

    sma50 = close.rolling(50, min_periods=20).mean()
    sma50_prev = sma50.shift(10)
    out["sma50_slope"] = (sma50 - sma50_prev) / sma50_prev

    out["rsi_norm"] = (_rsi(close) - 50.0) / 50.0

    returns = close.pct_change()
    out["volatility"] = returns.rolling(20, min_periods=10).std()

    out["momentum_10d"] = close.pct_change(10)

    if "volume" in df.columns:
        vol = df["volume"]
        vol_mean = vol.rolling(20, min_periods=10).mean()
        vol_std = vol.rolling(20, min_periods=10).std().replace(0, np.nan)
        out["volume_zscore"] = (vol - vol_mean) / vol_std
    else:
        out["volume_zscore"] = 0.0

    if {"open", "high", "low"}.issubset(df.columns):
        out["hammer_signal"] = detect_bullish_hammer(df).astype(float)
        out["engulfing_signal"] = detect_bullish_engulfing(df).astype(float)
        out["morning_star_signal"] = detect_morning_star(df).astype(float)
        out["doji_signal"] = detect_indecision_doji(df).astype(float)
    else:
        for col in PATTERN_COLUMNS:
            out[col] = 0.0

    out["close"] = close
    out["rolling_high"] = rolling_high
    out["sma50"] = sma50

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    return out


FEATURE_COLUMNS = (["drop_from_high", "sma20_dist", "sma50_slope", "rsi_norm",
                      "volatility", "momentum_10d", "volume_zscore"] + PATTERN_COLUMNS)


def classify_opportunity(row: pd.Series):
    """Devuelve (es_oportunidad, riesgo) según reglas de caída + tendencia.

    riesgo in {"alto", "bajo", None}. None si ni siquiera cumple la caída mínima.
    """
    drop = row["drop_from_high"]
    uptrend = row["sma50_slope"] > 0

    if drop < config.MIN_DROP_TO_CONSIDER:
        return False, None

    lo, hi = config.HIGH_RISK_DROP_RANGE
    if lo <= drop < hi:
        return True, "alto"

    if drop >= config.LOW_RISK_MIN_DROP:
        if config.LOW_RISK_REQUIRES_UPTREND and not uptrend:
            # caída grande pero sin señal de tendencia de fondo positiva:
            # se sigue considerando, pero como riesgo alto (más incertidumbre)
            return True, "alto"
        return True, "bajo"

    # cae entre HIGH_RISK_DROP_RANGE.hi y LOW_RISK_MIN_DROP -> zona intermedia,
    # se cataloga según tendencia
    return True, ("bajo" if uptrend else "alto")
