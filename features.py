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

    out["close"] = close
    out["rolling_high"] = rolling_high
    out["sma50"] = sma50

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    return out


FEATURE_COLUMNS = ["drop_from_high", "sma20_dist", "sma50_slope",
                    "rsi_norm", "volatility", "momentum_10d", "volume_zscore"]


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
