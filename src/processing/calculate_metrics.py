# ============================================================
# calculate_metrics.py — Motor de cálculo de riesgo
# ============================================================
# Contexto de negocio: Este módulo replica el trabajo que
# realiza un equipo de Risk Management cada mañana antes
# de la apertura del mercado. Los cálculos implementados
# aquí son requisitos regulatorios de Basilea III y métricas
# estándar de la industria de gestión de activos.
#
# Flujo del proceso:
# 1. Extrae retornos históricos desde PostgreSQL
# 2. Calcula VaR por 3 métodos (histórico, paramétrico, Monte Carlo)
# 3. Calcula métricas de rendimiento del portafolio
# 4. Persiste resultados en portfolio_metrics y var_history
# 5. Registra alertas si se superan umbrales críticos
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, date
from scipy import stats
from sqlalchemy import text
from src.database import get_engine

# ── Parámetros de riesgo ───────────────────────────────────
# Estos parámetros los define el Risk Manager o el Comité
# de Riesgo de la institución. Los dejamos como constantes
# configurables, no hardcodeadas dentro de las funciones.

RISK_FREE_RATE    = 0.0525   # Tasa libre de riesgo (T-Bill USA a 1 año, aprox.)
TRADING_DAYS      = 252      # Días de trading al año (estándar global)
LOOKBACK_DAYS     = 252      # Ventana histórica para cálculos de riesgo (1 año)
CONFIDENCE_95     = 0.95     # Nivel de confianza estándar regulatorio
CONFIDENCE_99     = 0.99     # Nivel de confianza para stress testing
MONTE_CARLO_SIMS  = 10000    # Simulaciones Monte Carlo (estándar industria)
VAR_THRESHOLD_PCT = float(os.getenv("VAR_THRESHOLD", "4.0"))  # Umbral de alerta (%)
PORTFOLIO_VALUE   = float(os.getenv("PORTFOLIO_VALUE", "100000"))  # Valor del portafolio USD
# ───────────────────────────────────────────────────────────


# ============================================================
# SECCIÓN 1: EXTRACCIÓN DE DATOS
# ============================================================

def get_portfolio_returns(lookback_days=LOOKBACK_DAYS):
    """
    Extrae retornos diarios del portafolio desde PostgreSQL.

    El portafolio se construye como un promedio equiponderado
    de los activos no-benchmark. En un fondo real, cada activo
    tendría un peso específico definido por el gestor. Para
    este sistema usamos pesos iguales — una decisión válida
    y transparente que debe mencionarse en el README.
    """
    engine = get_engine()

    query = text("""
        SELECT
            p.price_date,
            a.ticker,
            a.is_benchmark,
            p.daily_return,
            p.close_price
        FROM prices_daily p
        JOIN assets a ON p.asset_id = a.asset_id
        WHERE p.price_date >= CURRENT_DATE - INTERVAL ':days days'
          AND p.daily_return IS NOT NULL
          AND p.daily_return != 0
        ORDER BY p.price_date, a.ticker
    """.replace(":days", str(lookback_days)))

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        raise ValueError("No hay datos de retornos en la base de datos. "
                         "Ejecuta primero fetch_prices.py")

    # Separar benchmark (SPY) del portafolio
    benchmark_df  = df[df["is_benchmark"] == True].copy()
    portfolio_df  = df[df["is_benchmark"] == False].copy()

    # Retorno diario del portafolio = promedio equiponderado
    portfolio_returns = (
        portfolio_df
        .groupby("price_date")["daily_return"]
        .mean()
        .rename("portfolio_return")
    )

    # Retorno diario del benchmark (SPY)
    benchmark_returns = (
        benchmark_df
        .groupby("price_date")["daily_return"]
        .first()
        .rename("benchmark_return")
    )

    # Unir en un solo DataFrame alineado por fecha
    returns_df = pd.DataFrame({
        "portfolio_return": portfolio_returns,
        "benchmark_return": benchmark_returns
    }).dropna()

    print(f"📊 Datos cargados: {len(returns_df)} días de retornos "
          f"({returns_df.index.min()} → {returns_df.index.max()})")

    return returns_df


# ============================================================
# SECCIÓN 2: VALUE AT RISK (VaR)
# ============================================================

def calculate_var_historical(returns, confidence=CONFIDENCE_95):
    """
    Método 1: VaR Histórico

    Usa la distribución real de retornos observados sin
    asumir ninguna distribución estadística. Es el método
    más transparente y directo: "en el X% de los peores
    días históricos, perdimos al menos este porcentaje."

    Reguladores como la EBA (Europa) y la Fed (USA) aceptan
    este método como válido para el cálculo de capital mínimo.
    """
    var = np.percentile(returns, (1 - confidence) * 100)
    return abs(var)


def calculate_var_parametric(returns, confidence=CONFIDENCE_95):
    """
    Método 2: VaR Paramétrico (Varianza-Covarianza)

    Asume que los retornos siguen una distribución normal.
    Usa la media y desviación estándar de los retornos para
    calcular el VaR mediante la función de distribución normal.

    Es el método más rápido computacionalmente y el preferido
    en reportes regulatorios diarios por su reproducibilidad.
    Su debilidad: subestima el riesgo en eventos extremos
    (fat tails), por eso siempre se complementa con el método
    histórico y Monte Carlo.
    """
    mu    = np.mean(returns)
    sigma = np.std(returns)
    var   = stats.norm.ppf(1 - confidence, mu, sigma)
    return abs(var)


def calculate_var_montecarlo(returns, confidence=CONFIDENCE_95,
                              n_simulations=MONTE_CARLO_SIMS):
    """
    Método 3: VaR Monte Carlo

    Simula N escenarios futuros posibles usando la distribución
    estadística de los retornos históricos. Es el método más
    robusto porque captura mejor los eventos extremos y no
    depende de supuestos de normalidad.

    10,000 simulaciones es el estándar mínimo en la industria.
    Bancos de inversión como Goldman Sachs o JPMorgan corren
    100,000+ simulaciones en sus sistemas de producción.
    """
    mu    = np.mean(returns)
    sigma = np.std(returns)

    # Generamos N retornos simulados con la misma distribución
    simulated_returns = np.random.normal(mu, sigma, n_simulations)

    var = np.percentile(simulated_returns, (1 - confidence) * 100)
    return abs(var)


def calculate_all_var(returns):
    """
    Calcula el VaR con los tres métodos al 95% y 99%.
    Retorna un diccionario con todos los valores para
    persistir en la tabla var_history.
    """
    results = {
        "var_historical_95":  calculate_var_historical(returns, 0.95),
        "var_historical_99":  calculate_var_historical(returns, 0.99),
        "var_parametric_95":  calculate_var_parametric(returns, 0.95),
        "var_parametric_99":  calculate_var_parametric(returns, 0.99),
        "var_montecarlo_95":  calculate_var_montecarlo(returns, 0.95),
        "var_montecarlo_99":  calculate_var_montecarlo(returns, 0.99),
    }

    # Pérdida en USD sobre el valor total del portafolio
    results["var_amount_usd"] = float(results["var_historical_95"] * PORTFOLIO_VALUE)

    # Flag de alerta: ¿superó el umbral definido por el Risk Manager?
    results["threshold_breached"] = bool(
        results["var_historical_95"] * 100 > VAR_THRESHOLD_PCT
    )

    # Convertir todos los valores numpy a tipos nativos Python
    # psycopg2 no sabe manejar numpy.float64 ni numpy.bool_ directamente
    results = {
        key: float(value) if isinstance(value, (float, int)) and key != "threshold_breached"
        else value
        for key, value in results.items()
    }
    results["threshold_breached"] = bool(results["threshold_breached"])

    return results


# ============================================================
# SECCIÓN 3: MÉTRICAS DE RENDIMIENTO DEL PORTAFOLIO
# ============================================================

def calculate_sharpe_ratio(portfolio_returns,
                            risk_free_rate=RISK_FREE_RATE,
                            trading_days=TRADING_DAYS):
    """
    Sharpe Ratio — Rentabilidad ajustada al riesgo.

    Responde la pregunta fundamental de cualquier inversor:
    "¿Estoy siendo compensado adecuadamente por el riesgo que tomo?"

    Interpretación estándar de la industria:
    < 1.0  → Rentabilidad insuficiente para el riesgo asumido
    1.0–2.0 → Aceptable
    > 2.0  → Buena gestión de riesgo/retorno
    > 3.0  → Excepcional (muy raro en mercados eficientes)

    Fórmula: (Retorno_portafolio - Tasa_libre_riesgo) / Volatilidad
    Anualizado multiplicando por √252 (días de trading)
    """
    daily_rf = risk_free_rate / trading_days

    excess_returns = portfolio_returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days)

    return round(sharpe, 4)


def calculate_sortino_ratio(portfolio_returns,
                             risk_free_rate=RISK_FREE_RATE,
                             trading_days=TRADING_DAYS):
    """
    Sortino Ratio — Variante del Sharpe que penaliza solo
    la volatilidad negativa (downside deviation).

    Más preciso que el Sharpe porque un portafolio con muchos
    días de subidas bruscas es penalizado por Sharpe (alta
    volatilidad = malo) pero no por Sortino (esa volatilidad
    es positiva = bienvenida). Muy usado en hedge funds.
    """
    daily_rf = risk_free_rate / trading_days

    excess_returns  = portfolio_returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]
    downside_std    = downside_returns.std()

    if downside_std == 0:
        return 0.0

    sortino = (excess_returns.mean() / downside_std) * np.sqrt(trading_days)
    return round(sortino, 4)


def calculate_max_drawdown(portfolio_returns):
    """
    Maximum Drawdown — La mayor caída desde un máximo histórico.

    Responde: "¿Cuánto habría perdido un inversor en el peor
    momento posible del período analizado?"

    Es la métrica de riesgo más intuitiva para clientes no
    técnicos. En un comité de dirección, el Drawdown comunica
    el riesgo mucho mejor que la desviación estándar.

    Un Drawdown del -20% significa que en algún punto del
    período, el portafolio valía un 20% menos que en su
    máximo anterior.
    """
    # Construcción de la curva de valor acumulado
    cumulative = (1 + portfolio_returns).cumprod()

    # Máximo histórico en cada punto del tiempo
    rolling_max = cumulative.expanding().max()

    # Drawdown en cada punto = caída desde el máximo
    drawdown = (cumulative - rolling_max) / rolling_max

    max_dd = drawdown.min()
    return round(max_dd, 4)


def calculate_beta(portfolio_returns, benchmark_returns):
    """
    Beta — Sensibilidad del portafolio al movimiento del mercado.

    Responde: "Cuando el S&P 500 sube o baja un 1%,
    ¿cuánto sube o baja nuestro portafolio?"

    Interpretación:
    Beta = 1.0  → El portafolio se mueve igual que el mercado
    Beta > 1.0  → Más agresivo que el mercado (más riesgo/retorno)
    Beta < 1.0  → Más defensivo (menos riesgo/retorno)
    Beta < 0    → Se mueve en sentido opuesto al mercado (cobertura)

    En gestión de portafolios, ajustar el Beta es la herramienta
    principal para controlar la exposición al riesgo de mercado.
    """
    # Alineamos las series por fecha
    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns
    }).dropna()

    covariance = np.cov(aligned["portfolio"], aligned["benchmark"])[0][1]
    variance   = np.var(aligned["benchmark"])

    if variance == 0:
        return 1.0

    beta = covariance / variance
    return round(beta, 4)


def calculate_alpha(portfolio_returns, benchmark_returns,
                    risk_free_rate=RISK_FREE_RATE,
                    trading_days=TRADING_DAYS):
    """
    Alpha — Rentabilidad generada por encima del benchmark.

    Responde la pregunta más importante en gestión activa:
    "¿Estamos añadiendo valor real o simplemente siguiendo
    al mercado?"

    Alpha positivo = el gestor añade valor (justifica sus fees)
    Alpha negativo = mejor invertir en un ETF pasivo
    Alpha = 0      = el portafolio replica al benchmark

    Esta es la métrica por la que los gestores de fondos
    cobran sus comisiones de gestión. Un Alpha consistente
    y positivo es extraordinariamente difícil de mantener.
    """
    beta = calculate_beta(portfolio_returns, benchmark_returns)

    # Retornos anualizados
    portfolio_annual = portfolio_returns.mean() * trading_days
    benchmark_annual = benchmark_returns.mean() * trading_days

    # Fórmula CAPM: Alpha = R_p - [R_f + Beta * (R_m - R_f)]
    alpha = portfolio_annual - (
        risk_free_rate + beta * (benchmark_annual - risk_free_rate)
    )

    return round(alpha, 4)


def calculate_volatility(portfolio_returns, trading_days=TRADING_DAYS):
    """
    Volatilidad anualizada — desviación estándar de retornos.

    Medida base del riesgo de mercado. Todos los demás
    indicadores de riesgo (VaR, Sharpe, Beta) dependen
    de este cálculo. Se anualiza multiplicando por √252
    para que sea comparable entre períodos.
    """
    volatility = portfolio_returns.std() * np.sqrt(trading_days)
    return round(volatility, 4)


def calculate_cumulative_return(portfolio_returns):
    """
    Retorno acumulado del período analizado.
    Cuánto ha ganado o perdido el portafolio en total
    durante la ventana de análisis.
    """
    cumulative = (1 + portfolio_returns).prod() - 1
    return round(cumulative, 4)


# ============================================================
# SECCIÓN 4: PERSISTENCIA DE RESULTADOS
# ============================================================

def save_var_to_db(var_results, calculation_date):
    """
    Persiste el cálculo de VaR del día en var_history.
    Idempotente: si ya existe el registro del día, lo actualiza.
    """
    engine = get_engine()

    query = text("""
        INSERT INTO var_history (
            calculation_date, portfolio_value,
            var_historical_95, var_historical_99,
            var_parametric_95, var_parametric_99,
            var_montecarlo_95, var_montecarlo_99,
            var_amount_usd, threshold_breached
        ) VALUES (
            :calculation_date, :portfolio_value,
            :var_historical_95, :var_historical_99,
            :var_parametric_95, :var_parametric_99,
            :var_montecarlo_95, :var_montecarlo_99,
            :var_amount_usd, :threshold_breached
        )
        ON CONFLICT (calculation_date)
        DO UPDATE SET
            var_historical_95  = EXCLUDED.var_historical_95,
            var_historical_99  = EXCLUDED.var_historical_99,
            var_parametric_95  = EXCLUDED.var_parametric_95,
            var_parametric_99  = EXCLUDED.var_parametric_99,
            var_montecarlo_95  = EXCLUDED.var_montecarlo_95,
            var_montecarlo_99  = EXCLUDED.var_montecarlo_99,
            var_amount_usd     = EXCLUDED.var_amount_usd,
            threshold_breached = EXCLUDED.threshold_breached
    """)

    with engine.connect() as conn:
        conn.execute(query, {
            "calculation_date":  calculation_date,
            "portfolio_value":   PORTFOLIO_VALUE,
            **var_results
        })
        conn.commit()

    print(f"   💾 VaR guardado para {calculation_date}")


def save_metrics_to_db(metrics, calculation_date):
    """
    Persiste las métricas de rendimiento del día en
    portfolio_metrics. Idempotente como save_var_to_db.
    """
    engine = get_engine()

    query = text("""
        INSERT INTO portfolio_metrics (
            calculation_date, portfolio_value, daily_return,
            cumulative_return, sharpe_ratio, sortino_ratio,
            max_drawdown, beta, alpha, volatility_30d
        ) VALUES (
            :calculation_date, :portfolio_value, :daily_return,
            :cumulative_return, :sharpe_ratio, :sortino_ratio,
            :max_drawdown, :beta, :alpha, :volatility_30d
        )
        ON CONFLICT (calculation_date)
        DO UPDATE SET
            sharpe_ratio      = EXCLUDED.sharpe_ratio,
            sortino_ratio     = EXCLUDED.sortino_ratio,
            max_drawdown      = EXCLUDED.max_drawdown,
            beta              = EXCLUDED.beta,
            alpha             = EXCLUDED.alpha,
            volatility_30d    = EXCLUDED.volatility_30d,
            cumulative_return = EXCLUDED.cumulative_return
    """)

    with engine.connect() as conn:
        conn.execute(query, {
            "calculation_date": calculation_date,
            "portfolio_value":  PORTFOLIO_VALUE,
            **metrics
        })
        conn.commit()

    print(f"   💾 Métricas guardadas para {calculation_date}")


def save_alert_to_db(var_results, var_metrics):
    """
    Registra una alerta en risk_alerts si el VaR
    superó el umbral definido por el Risk Manager.

    Este registro es el insumo que usará el módulo
    de notificaciones para enviar el email ejecutivo.
    """
    if not var_results.get("threshold_breached"):
        return

    engine  = get_engine()
    var_pct = var_results["var_historical_95"] * 100

    severity = "MEDIUM"
    if var_pct > VAR_THRESHOLD_PCT * 1.5:
        severity = "HIGH"
    if var_pct > VAR_THRESHOLD_PCT * 2.0:
        severity = "CRITICAL"

    message = (
        f"VaR histórico al 95% alcanzó {var_pct:.2f}%, "
        f"superando el umbral de {VAR_THRESHOLD_PCT:.1f}%. "
        f"Pérdida máxima estimada: "
        f"${var_results['var_amount_usd']:,.0f} USD sobre "
        f"un portafolio de ${PORTFOLIO_VALUE:,.0f} USD."
    )

    query = text("""
        INSERT INTO risk_alerts (
            alert_type, severity, metric_name,
            metric_value, threshold_value, message
        ) VALUES (
            :alert_type, :severity, :metric_name,
            :metric_value, :threshold_value, :message
        )
    """)

    with engine.connect() as conn:
        conn.execute(query, {
            "alert_type":      "VAR_BREACH",
            "severity":        severity,
            "metric_name":     "var_historical_95",
            "metric_value":    var_results["var_historical_95"],
            "threshold_value": VAR_THRESHOLD_PCT / 100,
            "message":         message
        })
        conn.commit()

    print(f"   🚨 Alerta {severity} registrada — "
          f"VaR: {var_pct:.2f}% > umbral: {VAR_THRESHOLD_PCT}%")


# ============================================================
# SECCIÓN 5: PIPELINE PRINCIPAL
# ============================================================

def run_metrics():
    """
    Pipeline principal de cálculo de métricas.

    Orquesta el flujo completo de Risk Management diario:
    1. Extrae retornos históricos del último año
    2. Calcula VaR con 3 métodos regulatorios
    3. Calcula métricas de rendimiento del portafolio
    4. Persiste resultados en PostgreSQL
    5. Registra alertas si se superan umbrales
    """
    print("=" * 55)
    print("  PORTFOLIO RISK SYSTEM — Cálculo de Métricas")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    calculation_date = date.today()

    # ── 1. Extraer datos ────────────────────────────────────
    print("\n[1/4] Extrayendo retornos históricos...")
    returns_df = get_portfolio_returns()

    portfolio_returns = returns_df["portfolio_return"]
    benchmark_returns = returns_df["benchmark_return"]

    # ── 2. Calcular VaR ─────────────────────────────────────
    print("\n[2/4] Calculando Value at Risk...")

    var_results = calculate_all_var(portfolio_returns.values)

    print(f"\n   VaR Histórico  95%: {var_results['var_historical_95']*100:.2f}%  "
          f"99%: {var_results['var_historical_99']*100:.2f}%")
    print(f"   VaR Paramétrico 95%: {var_results['var_parametric_95']*100:.2f}%  "
          f"99%: {var_results['var_parametric_99']*100:.2f}%")
    print(f"   VaR Monte Carlo 95%: {var_results['var_montecarlo_95']*100:.2f}%  "
          f"99%: {var_results['var_montecarlo_99']*100:.2f}%")
    print(f"   Pérdida máx. est.:  ${var_results['var_amount_usd']:,.0f} USD")
    print(f"   Umbral superado:    {'🚨 SÍ' if var_results['threshold_breached'] else '✅ NO'}")

    # ── 3. Calcular métricas de rendimiento ─────────────────
    print("\n[3/4] Calculando métricas de rendimiento...")

    metrics = {
        "daily_return":      round(float(portfolio_returns.iloc[-1]), 6),
        "cumulative_return": float(calculate_cumulative_return(portfolio_returns)),
        "sharpe_ratio":      float(calculate_sharpe_ratio(portfolio_returns)),
        "sortino_ratio":     float(calculate_sortino_ratio(portfolio_returns)),
        "max_drawdown":      float(calculate_max_drawdown(portfolio_returns)),
        "beta":              float(calculate_beta(portfolio_returns, benchmark_returns)),
        "alpha":             float(calculate_alpha(portfolio_returns, benchmark_returns)),
        "volatility_30d":    float(calculate_volatility(portfolio_returns)),
    }

    print(f"\n   Sharpe Ratio:      {metrics['sharpe_ratio']}")
    print(f"   Sortino Ratio:     {metrics['sortino_ratio']}")
    print(f"   Max Drawdown:      {metrics['max_drawdown']*100:.2f}%")
    print(f"   Beta vs S&P 500:   {metrics['beta']}")
    print(f"   Alpha anualizado:  {metrics['alpha']*100:.2f}%")
    print(f"   Volatilidad 30d:   {metrics['volatility_30d']*100:.2f}%")
    print(f"   Retorno acumulado: {metrics['cumulative_return']*100:.2f}%")

    # ── 4. Persistir en PostgreSQL ───────────────────────────
    print("\n[4/4] Persistiendo resultados en PostgreSQL...")

    save_var_to_db(var_results, calculation_date)
    save_metrics_to_db(metrics, calculation_date)
    save_alert_to_db(var_results, metrics)

    # ── Resumen ejecutivo ────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESUMEN EJECUTIVO")
    print("=" * 55)
    print(f"  Estado del portafolio : "
          f"{'🚨 EN ALERTA' if var_results['threshold_breached'] else '✅ DENTRO DE LÍMITES'}")
    print(f"  VaR del día (95%)     : "
          f"{var_results['var_historical_95']*100:.2f}%")
    print(f"  Pérdida máx. est.     : "
          f"${var_results['var_amount_usd']:,.0f} USD")
    print(f"  Sharpe Ratio          : {metrics['sharpe_ratio']}")
    print(f"  Max Drawdown          : {metrics['max_drawdown']*100:.2f}%")
    print("=" * 55)

    return var_results, metrics


if __name__ == "__main__":
    run_metrics()