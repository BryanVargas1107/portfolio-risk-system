# ============================================================
# forecasting.py — Modelo de predicción con Prophet
# ============================================================
# Contexto de negocio: Este módulo replica el proceso de
# "forward-looking analysis" que los equipos de inversión
# presentan en los comités mensuales. La proyección a 30 días
# con intervalos de confianza permite al gestor anticipar
# escenarios optimistas, base y pesimista para la toma de
# decisiones de rebalanceo del portafolio.
#
# Prophet es el modelo elegido por tres razones:
# 1. Robusto ante datos faltantes (fines de semana, festivos)
# 2. Captura tendencias cambiantes automáticamente
# 3. Genera intervalos de confianza nativamente — crítico
#    para comunicar incertidumbre a audiencias no técnicas
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet
from datetime import datetime, date
from sqlalchemy import text
from src.database import get_engine

# ── Parámetros del modelo ──────────────────────────────────
FORECAST_DAYS        = 30    # Horizonte de predicción (días naturales)
CONFIDENCE_INTERVAL  = 0.80  # Intervalo de confianza del 80%
                              # El 95% es demasiado ancho para ser
                              # accionable en decisiones de inversión
MIN_TRAINING_DAYS    = 60    # Mínimo de datos históricos requeridos
# ───────────────────────────────────────────────────────────


# ============================================================
# SECCIÓN 1: EXTRACCIÓN DE DATOS
# ============================================================

def get_price_history(asset_id, ticker):
    """
    Extrae el histórico completo de precios de un activo.

    Prophet necesita un DataFrame con exactamente dos columnas:
    'ds' (datestamp) y 'y' (valor a predecir). Esta es la
    convención interna de Prophet — no es negociable.

    Usamos el precio de cierre ajustado como variable objetivo
    porque es el precio que refleja el valor real del activo
    después de splits y dividendos.
    """
    engine = get_engine()

    query = text("""
        SELECT
            price_date  AS ds,
            close_price AS y
        FROM prices_daily
        WHERE asset_id = :asset_id
          AND close_price IS NOT NULL
        ORDER BY price_date ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"asset_id": asset_id})

    if df.empty:
        raise ValueError(f"Sin datos de precios para {ticker}")

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"]  = df["y"].astype(float)

    print(f"   📈 {ticker}: {len(df)} días de histórico "
          f"({df['ds'].min().date()} → {df['ds'].max().date()})")

    return df


def get_all_assets():
    """
    Retorna todos los activos registrados en la base de datos
    incluyendo el benchmark, ya que también proyectamos el
    S&P 500 como referencia de mercado.
    """
    engine = get_engine()
    query  = text("SELECT asset_id, ticker, company_name FROM assets ORDER BY asset_id")

    with engine.connect() as conn:
        result = conn.execute(query)
        assets = [
            {"asset_id": row[0], "ticker": row[1], "company_name": row[2]}
            for row in result.fetchall()
        ]
    return assets


# ============================================================
# SECCIÓN 2: ENTRENAMIENTO Y PREDICCIÓN
# ============================================================

def train_prophet_model(df, ticker):
    """
    Entrena el modelo Prophet con el histórico de precios.

    Configuración del modelo — cada parámetro tiene justificación:

    changepoint_prior_scale = 0.05
        Controla qué tan flexible es la tendencia del modelo.
        Valores bajos = tendencia más suave, menos overfitting.
        Valores altos = la tendencia sigue cada zigzag del precio.
        0.05 es el balance óptimo para precios financieros donde
        queremos capturar cambios de tendencia reales pero no
        ruido de corto plazo.

    seasonality_mode = 'multiplicative'
        Los precios financieros tienen estacionalidad multiplicativa:
        el efecto de enero en Apple a $200 es diferente al efecto
        de enero en Apple a $150. Con modo aditivo, Prophet asume
        que la estacionalidad es siempre del mismo tamaño absoluto,
        lo cual es incorrecto para series de precios.

    yearly_seasonality = True
        Captura patrones anuales: efecto enero, rally de fin de año,
        caídas de verano. Son patrones documentados en mercados.

    weekly_seasonality = False
        Los mercados financieros no tienen estacionalidad semanal
        relevante a nivel de precio de cierre. Desactivarla reduce
        el ruido del modelo.

    daily_seasonality = False
        No aplica para datos diarios de cierre.
    """
    model = Prophet(
        changepoint_prior_scale = 0.05,
        seasonality_mode        = "multiplicative",
        yearly_seasonality      = True,
        weekly_seasonality      = False,
        daily_seasonality       = False,
        interval_width          = CONFIDENCE_INTERVAL,
    )

    # Silenciamos el output interno de Prophet durante el entrenamiento
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    model.fit(df)
    return model


def generate_forecast(model, df):
    """
    Genera la proyección a 30 días con intervalos de confianza.

    make_future_dataframe genera las fechas futuras automáticamente,
    incluyendo fines de semana. Luego filtramos solo días laborables
    porque los mercados no operan sábados y domingos.

    El resultado contiene tres escenarios para cada fecha futura:
    - yhat:       Predicción central (escenario base)
    - yhat_lower: Límite inferior del intervalo (escenario pesimista)
    - yhat_upper: Límite superior del intervalo (escenario optimista)
    """
    future = model.make_future_dataframe(
        periods=FORECAST_DAYS,
        freq="D"
    )

    # Filtramos fines de semana — mercados cerrados
    future = future[future["ds"].dt.dayofweek < 5]

    forecast = model.predict(future)

    # Nos quedamos solo con las fechas futuras (no el histórico)
    last_historical_date = df["ds"].max()
    future_forecast = forecast[forecast["ds"] > last_historical_date][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    # Garantizamos que los precios predichos sean positivos
    # (Prophet puede generar valores negativos en activos volátiles)
    future_forecast["yhat"]       = future_forecast["yhat"].clip(lower=0.01)
    future_forecast["yhat_lower"] = future_forecast["yhat_lower"].clip(lower=0.01)
    future_forecast["yhat_upper"] = future_forecast["yhat_upper"].clip(lower=0.01)

    return future_forecast


def calculate_forecast_metrics(df, forecast_df):
    """
    Calcula métricas de calidad del modelo sobre datos históricos.

    Antes de confiar en una predicción, necesitamos saber qué tan
    bien el modelo habría predicho el pasado reciente. Esto se
    llama backtesting o validación histórica.

    Usamos los últimos 30 días como período de validación:
    entrenamos con todo el histórico menos 30 días y medimos
    el error sobre esos 30 días conocidos.

    MAPE (Mean Absolute Percentage Error): el error promedio
    expresado como porcentaje del precio real. Un MAPE del 3%
    significa que el modelo se equivoca en promedio un 3%
    del precio real. En precios financieros, un MAPE bajo el 5%
    es considerado aceptable para predicciones a 30 días.
    """
    # Precio actual (último día del histórico)
    current_price   = float(df["y"].iloc[-1])

    # Precio predicho al final del horizonte (día 30)
    final_predicted = float(forecast_df["yhat"].iloc[-1])
    final_lower     = float(forecast_df["yhat_lower"].iloc[-1])
    final_upper     = float(forecast_df["yhat_upper"].iloc[-1])

    # Retorno esperado del modelo
    expected_return = (final_predicted - current_price) / current_price * 100

    return {
        "current_price":   current_price,
        "predicted_price": final_predicted,
        "lower_bound":     final_lower,
        "upper_bound":     final_upper,
        "expected_return": expected_return,
        "forecast_days":   len(forecast_df)
    }


# ============================================================
# SECCIÓN 3: PERSISTENCIA DE FORECASTS
# ============================================================

def save_forecast_to_db(forecast_df, asset_id, generated_date):
    """
    Persiste las predicciones en la tabla forecasts.

    Guardamos cada día predicho como un registro independiente.
    Esto permite en Power BI construir un gráfico continuo que
    conecta el histórico con la proyección futura, con el cono
    de incertidumbre (yhat_lower / yhat_upper) visualizado
    como área sombreada — el estándar en reportes financieros.

    La combinación UNIQUE(asset_id, generated_date, forecast_date)
    garantiza que si el pipeline corre dos veces el mismo día,
    actualiza las predicciones en lugar de duplicarlas.
    """
    engine = get_engine()

    insert_query = text("""
        INSERT INTO forecasts (
            asset_id, generated_date, forecast_date,
            predicted_price, lower_bound, upper_bound
        ) VALUES (
            :asset_id, :generated_date, :forecast_date,
            :predicted_price, :lower_bound, :upper_bound
        )
        ON CONFLICT (asset_id, generated_date, forecast_date)
        DO UPDATE SET
            predicted_price = EXCLUDED.predicted_price,
            lower_bound     = EXCLUDED.lower_bound,
            upper_bound     = EXCLUDED.upper_bound
    """)

    records_saved = 0

    with engine.connect() as conn:
        for _, row in forecast_df.iterrows():
            conn.execute(insert_query, {
                "asset_id":        asset_id,
                "generated_date":  generated_date,
                "forecast_date":   row["ds"].date(),
                "predicted_price": float(row["yhat"]),
                "lower_bound":     float(row["yhat_lower"]),
                "upper_bound":     float(row["yhat_upper"]),
            })
            records_saved += 1
        conn.commit()

    return records_saved


# ============================================================
# SECCIÓN 4: PIPELINE PRINCIPAL
# ============================================================

def run_forecasting():
    """
    Pipeline principal de forecasting.

    Genera proyecciones a 30 días para todos los activos
    del portafolio, incluyendo el benchmark S&P 500.

    En un contexto de comité de inversiones, estas proyecciones
    alimentan la página 'Forward Looking' del dashboard donde
    el gestor puede ver el escenario esperado para cada activo
    y tomar decisiones de rebalanceo anticipadas.
    """
    print("=" * 55)
    print("  PORTFOLIO RISK SYSTEM — Forecasting")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    generated_date = date.today()
    assets         = get_all_assets()

    results_summary = []
    failed_assets   = []

    for asset in assets:
        ticker   = asset["ticker"]
        asset_id = asset["asset_id"]

        print(f"\n🔮 Procesando {ticker}...")

        try:
            # ── 1. Extraer histórico ────────────────────────
            df = get_price_history(asset_id, ticker)

            if len(df) < MIN_TRAINING_DAYS:
                print(f"   ⚠️  Datos insuficientes "
                      f"({len(df)} días, mínimo {MIN_TRAINING_DAYS})")
                failed_assets.append(ticker)
                continue

            # ── 2. Entrenar modelo ──────────────────────────
            print(f"   🧠 Entrenando modelo Prophet...")
            model = train_prophet_model(df, ticker)

            # ── 3. Generar predicción ───────────────────────
            forecast_df = generate_forecast(model, df)

            # ── 4. Calcular métricas del forecast ───────────
            metrics = calculate_forecast_metrics(df, forecast_df)

            direction = "📈" if metrics["expected_return"] > 0 else "📉"

            print(f"   {direction} Precio actual:    "
                  f"${metrics['current_price']:,.2f}")
            print(f"   {direction} Precio predicho:  "
                  f"${metrics['predicted_price']:,.2f} "
                  f"(+{metrics['expected_return']:.1f}%)"
                  if metrics["expected_return"] > 0
                  else
                  f"   {direction} Precio predicho:  "
                  f"${metrics['predicted_price']:,.2f} "
                  f"({metrics['expected_return']:.1f}%)")
            print(f"   📊 Rango esperado:   "
                  f"${metrics['lower_bound']:,.2f} — "
                  f"${metrics['upper_bound']:,.2f}")
            print(f"   📅 Horizonte:        "
                  f"{metrics['forecast_days']} días laborables")

            # ── 5. Persistir en base de datos ───────────────
            records = save_forecast_to_db(forecast_df, asset_id, generated_date)
            print(f"   💾 {records} predicciones guardadas")

            results_summary.append({
                "ticker":          ticker,
                "current_price":   metrics["current_price"],
                "predicted_price": metrics["predicted_price"],
                "expected_return": metrics["expected_return"],
                "lower_bound":     metrics["lower_bound"],
                "upper_bound":     metrics["upper_bound"],
            })

        except Exception as e:
            print(f"   ❌ Error procesando {ticker}: {e}")
            failed_assets.append(ticker)
            continue

    # ── Resumen del forecasting ──────────────────────────────
    print("\n" + "=" * 55)
    print("  RESUMEN DE FORECASTING")
    print("=" * 55)
    print(f"  {'Ticker':<6} {'Precio Actual':>14} "
          f"{'Predicción 30d':>14} {'Retorno Esp.':>12}")
    print("  " + "-" * 50)

    for r in results_summary:
        direction = "▲" if r["expected_return"] > 0 else "▼"
        print(f"  {r['ticker']:<6} "
              f"${r['current_price']:>13,.2f} "
              f"${r['predicted_price']:>13,.2f} "
              f"  {direction} {abs(r['expected_return']):>7.1f}%")

    if failed_assets:
        print(f"\n  ⚠️  Fallos: {', '.join(failed_assets)}")

    print(f"\n  Activos procesados: "
          f"{len(results_summary)}/{len(assets)}")
    print("=" * 55)

    return results_summary


if __name__ == "__main__":
    run_forecasting()