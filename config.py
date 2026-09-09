"""
Configuración central del sistema de agentes evolutivos de trading.
"""
import os

# ---------------------------------------------------------------------------
# Universo de activos
# ---------------------------------------------------------------------------
TICKERS = {
     "acciones": [
        # Tecnología / IA
        "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL",
        "AMZN", "META", "AVGO", "ORCL", "CRM",
        "AMD", "INTC", "QCOM", "MU", "TSM",

        # Automotriz
        "TSLA", "F", "GM",

        # Finanzas
        "JPM", "BAC", "WFC", "GS", "MS",
        "V", "MA", "AXP", "BRK-B",

        # Salud
        "LLY", "JNJ", "UNH", "MRK", "ABBV",
        "PFE", "AMGN",

        # Consumo
        "COST", "WMT", "HD", "MCD", "NKE",
        "KO", "PEP",

        # Comunicación
        "NFLX", "DIS", "CMCSA",

        # Energía
        "XOM", "CVX", "COP",

        # Industria
        "CAT", "GE", "HON", "BA",

        # Growth / software
        "PLTR", "CRWD", "PANW"
    ],

    "mercado": [
        "^GSPC",
        "^DJI",
        "^IXIC",
        "^RUT",
        "^VIX",

        "SPY",
        "QQQ",
        "DIA",
        "IWM",

        "XLK",
        "XLF",
        "XLV",
        "XLE",
        "XLY",
        "XLP",
        "XLI",
        "XLC",
        "XLU",
        "XLRE",
        "XLB"
    ],

    "cripto": [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "BNB-USD",
        "XRP-USD",
        "ADA-USD",
        "DOGE-USD",
        "AVAX-USD",
        "LINK-USD",
        "DOT-USD",
        "TRX-USD",
        "LTC-USD",
        "BCH-USD",
        "UNI-USD",
        "ATOM-USD"
    ],

    "commodities": [
        "GC=F",
        "SI=F",
        "CL=F",
        "BZ=F",
        "NG=F",
        "HG=F"
    ],

    "bonos": [
        "^TNX",
        "^FVX",
        "^IRX",
        "TLT",
        "IEF",
        "SHY"
    ],

    "divisas": [
        "DX-Y.NYB",
        "EURUSD=X",
        "JPY=X",
        "GBPUSD=X",
        "MXN=X"
    ]
}

ALL_TICKERS = [t for grupo in TICKERS.values() for t in grupo]


def risk_group(ticker: str) -> str:
    for grupo, lista in TICKERS.items():
        if ticker in lista:
            return grupo
    return "otro"


# ---------------------------------------------------------------------------
# Rutas de datos y estado
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_cache")       # históricos descargados (csv)
STATE_DIR = os.path.join(BASE_DIR, "state")            # población + predicciones pendientes
OUTPUT_DIR = os.path.join(BASE_DIR, "output")           # reportes diarios (png/json)

for d in (DATA_DIR, STATE_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

POPULATION_FILE = os.path.join(STATE_DIR, "population.json")
PENDING_FILE = os.path.join(STATE_DIR, "pending_predictions.json")
HISTORY_LOG_FILE = os.path.join(STATE_DIR, "population_history.json")

# ---------------------------------------------------------------------------
# Datos históricos
# ---------------------------------------------------------------------------
YEARS_HISTORY = 5          # años de histórico diario a descargar
MIN_WARMUP_DAYS = 60        # días necesarios antes de poder calcular indicadores

# ---------------------------------------------------------------------------
# Población de agentes (evolución)
# ---------------------------------------------------------------------------
POPULATION_SIZE = 40
HIDDEN_SIZE = 8              # neuronas ocultas de cada agente
N_FEATURES = 8                # tamaño del vector de entrada (ver features.py)

INITIAL_HAPPINESS = 50.0
DEATH_HAPPINESS = 0.0        # el agente muere si su felicidad cae a esto o menos
MAX_HAPPINESS = 100.0

MUTATION_RATE = 0.25          # probabilidad de mutar cada peso
MUTATION_STRENGTH = 0.35      # magnitud de la mutación (ruido gaussiano)

EVAL_HORIZON_DAYS = 5          # días después de una compra sugerida para evaluarla
GAIN_SCALE = 400.0              # sensibilidad de la felicidad al % de ganancia/pérdida

# ---------------------------------------------------------------------------
# Reglas de oportunidad de compra / riesgo
# ---------------------------------------------------------------------------
# Una "oportunidad" solo existe si el precio cayó desde su máximo reciente (52
# semanas) al menos MIN_DROP_TO_CONSIDER. A partir de ahí:
#   RIESGO ALTO -> caída pequeña (poco descuento, mercado apenas corrigiendo)
#   RIESGO BAJO -> caída mayor PERO con tendencia de fondo positiva (SMA50 subiendo)
MIN_DROP_TO_CONSIDER = 0.03      # 3% mínimo de caída desde el máximo de 52 semanas

HIGH_RISK_DROP_RANGE = (0.03, 0.08)   # 3% - 8% de caída => riesgo alto
LOW_RISK_MIN_DROP = 0.08                # >= 8% de caída...
LOW_RISK_REQUIRES_UPTREND = True         # ...y SMA50 con pendiente positiva => riesgo bajo

BUY_PROB_THRESHOLD_BASE = 0.55    # umbral base de "creo que es compra" (cada agente lo muta)
