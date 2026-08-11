"""
Descarga y cachea el histórico diario de cada activo usando yfinance
(API gratuita, no requiere API key).
"""
import os
import time
import pandas as pd
import yfinance as yf

import config


def _cache_path(ticker: str) -> str:
    safe = ticker.replace("^", "IDX_")
    return os.path.join(config.DATA_DIR, f"{safe}.csv")


def download_ticker(ticker: str, years: int = config.YEARS_HISTORY,
                     retries: int = 3) -> pd.DataFrame:
    """Descarga histórico diario de un ticker. Reintenta en caso de fallo de red."""
    period = f"{years}y"
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval="1d",
                              auto_adjust=True, progress=False)
            if df is None or df.empty:
                raise ValueError(f"Sin datos para {ticker}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns=str.lower)
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            df.index.name = "date"
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"No se pudo descargar {ticker}: {last_err}")


def update_cache(ticker: str) -> pd.DataFrame:
    """Descarga el histórico completo y lo guarda/actualiza en cache local."""
    df = download_ticker(ticker)
    df.to_csv(_cache_path(ticker))
    return df


def load_cached(ticker: str) -> pd.DataFrame:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return update_cache(ticker)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def refresh_recent(ticker: str, max_age_days: int = 1) -> pd.DataFrame:
    """Si el cache existe pero está desactualizado, vuelve a descargar todo
    (yfinance es gratis y esto es lo más simple/robusto para un histórico diario)."""
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return update_cache(ticker)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    last_date = df.index.max()
    if (pd.Timestamp.now().normalize() - last_date.normalize()).days > max_age_days:
        return update_cache(ticker)
    return df


def load_all(tickers=None, refresh: bool = False) -> dict:
    """Devuelve {ticker: DataFrame} para todos los tickers configurados."""
    tickers = tickers or config.ALL_TICKERS
    data = {}
    for t in tickers:
        try:
            df = refresh_recent(t) if refresh else load_cached(t)
            data[t] = df
            print(f"[data_provider] {t}: {len(df)} filas ({df.index.min().date()} -> {df.index.max().date()})")
        except Exception as e:  # noqa: BLE001
            print(f"[data_provider] ERROR con {t}: {e}")
    return data


if __name__ == "__main__":
    load_all(refresh=True)
