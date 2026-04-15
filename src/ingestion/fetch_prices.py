# ============================================================
# fetch_prices.py — Ingesta de precios desde Yahoo Finance
# ============================================================
# Contexto de negocio: Este módulo replica el proceso de
# "market data ingestion" que ocurre cada día después del
# cierre del mercado (4:00 PM EST). Los sistemas de riesgo
# necesitan los precios del día para calcular el VaR y las
# métricas del portafolio antes de la apertura del día siguiente.
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from src.database import get_engine

# ── Configuración ──────────────────────────────────────────
# Período de carga inicial: 2 años de histórico.
# En producción, después del primer día solo se cargan
# los datos del día anterior (carga incremental).
HISTORICAL_YEARS = 2
# ───────────────────────────────────────────────────────────


def get_assets_from_db():
    """
    Lee los activos registrados en la base de datos.
    El sistema monitorea solo los activos autorizados —
    el mismo principio que una lista de instrumentos
    aprobados en una mesa de trading.
    """
    engine = get_engine()
    query = "SELECT asset_id, ticker, company_name FROM assets ORDER BY asset_id"
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        assets = [
            {"asset_id": row[0], "ticker": row[1], "company_name": row[2]}
            for row in result.fetchall()
        ]
    
    print(f"📋 {len(assets)} activos registrados en el sistema:")
    for asset in assets:
        print(f"   [{asset['asset_id']}] {asset['ticker']} — {asset['company_name']}")
    
    return assets


def download_prices(ticker, years=HISTORICAL_YEARS):
    """
    Descarga precios históricos desde Yahoo Finance.
    
    Yahoo Finance es una fuente de datos de mercado ampliamente
    usada para prototipos y análisis. En producción bancaria
    se usan proveedores como Bloomberg o Refinitiv, pero la
    lógica de procesamiento es idéntica.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"\n⬇️  Descargando {ticker}...")
    
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True   # ajusta precios por splits y dividendos
        )
        
        if df.empty:
            print(f"   ⚠️  Sin datos para {ticker}")
            return None
        
        # Renombramos columnas a snake_case para consistencia con el schema
        df = df.rename(columns={
            "Open":   "open_price",
            "High":   "high_price",
            "Low":    "low_price",
            "Close":  "close_price",
            "Volume": "volume"
        })
        
        # Limpieza del índice de fechas
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "price_date"
        df = df.reset_index()
        df["price_date"] = df["price_date"].dt.date
        
        # Precio ajustado = close (ya ajustado por yfinance con auto_adjust=True)
        df["adj_close"] = df["close_price"]
        
        print(f"   ✅ {len(df)} registros descargados "
              f"({df['price_date'].min()} → {df['price_date'].max()})")
        
        return df
    
    except Exception as e:
        print(f"   ❌ Error descargando {ticker}: {e}")
        return None


def calculate_daily_returns(df):
    """
    Calcula el retorno logarítmico diario.
    
    Usamos retornos logarítmicos (ln) en lugar de retornos
    simples porque son aditivos en el tiempo — propiedad
    fundamental para calcular el VaR histórico y otras
    métricas de riesgo. Es el estándar en la industria
    financiera y en los modelos de Basilea III.
    
    Fórmula: r_t = ln(P_t / P_{t-1})
    """
    df = df.sort_values("price_date").copy()
    df["daily_return"] = np.log(
        df["close_price"] / df["close_price"].shift(1)
    )
    # El primer registro no tiene retorno (no hay día anterior)
    df["daily_return"] = df["daily_return"].fillna(0)
    return df


def save_prices_to_db(df, asset_id, ticker):
    """
    Persiste los precios en PostgreSQL.
    
    Usamos INSERT ... ON CONFLICT DO NOTHING para que la
    operación sea idempotente — si el pipeline se ejecuta
    dos veces el mismo día, no duplica registros.
    Esto es crítico en sistemas de producción donde los
    procesos pueden reiniciarse por fallos.
    """
    engine = get_engine()
    
    records_inserted = 0
    records_skipped  = 0
    
    insert_query = text("""
        INSERT INTO prices_daily (
            asset_id, price_date, open_price, high_price,
            low_price, close_price, adj_close, volume, daily_return
        ) VALUES (
            :asset_id, :price_date, :open_price, :high_price,
            :low_price, :close_price, :adj_close, :volume, :daily_return
        )
        ON CONFLICT (asset_id, price_date) DO NOTHING
    """)
    
    with engine.connect() as conn:
        for _, row in df.iterrows():
            result = conn.execute(insert_query, {
                "asset_id":     asset_id,
                "price_date":   row["price_date"],
                "open_price":   float(row["open_price"])   if pd.notna(row["open_price"])   else None,
                "high_price":   float(row["high_price"])   if pd.notna(row["high_price"])   else None,
                "low_price":    float(row["low_price"])    if pd.notna(row["low_price"])     else None,
                "close_price":  float(row["close_price"])  if pd.notna(row["close_price"])  else None,
                "adj_close":    float(row["adj_close"])    if pd.notna(row["adj_close"])     else None,
                "volume":       int(row["volume"])         if pd.notna(row["volume"])        else None,
                "daily_return": float(row["daily_return"]) if pd.notna(row["daily_return"]) else None,
            })
            if result.rowcount > 0:
                records_inserted += 1
            else:
                records_skipped += 1
        conn.commit()
    
    print(f"   💾 {records_inserted} registros insertados, "
          f"{records_skipped} ya existían")
    
    return records_inserted


def run_ingestion():
    """
    Pipeline principal de ingesta.
    
    Orquesta el flujo completo:
    1. Lee activos autorizados desde la base de datos
    2. Descarga precios históricos para cada activo
    3. Calcula retornos logarítmicos
    4. Persiste en PostgreSQL con control de duplicados
    
    Este es el proceso que GitHub Actions ejecutará
    automáticamente cada día al cierre del mercado.
    """
    print("=" * 55)
    print("  PORTFOLIO RISK SYSTEM — Ingesta de Datos")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    assets = get_assets_from_db()
    
    total_inserted = 0
    failed_tickers = []
    
    for asset in assets:
        ticker   = asset["ticker"]
        asset_id = asset["asset_id"]
        
        # 1. Descargar precios
        df = download_prices(ticker)
        if df is None:
            failed_tickers.append(ticker)
            continue
        
        # 2. Calcular retornos logarítmicos
        df = calculate_daily_returns(df)
        
        # 3. Persistir en base de datos
        inserted = save_prices_to_db(df, asset_id, ticker)
        total_inserted += inserted
    
    # ── Resumen de ejecución ───────────────────────────────
    print("\n" + "=" * 55)
    print("  RESUMEN DE INGESTA")
    print("=" * 55)
    print(f"  Activos procesados : {len(assets) - len(failed_tickers)}/{len(assets)}")
    print(f"  Registros totales  : {total_inserted}")
    
    if failed_tickers:
        print(f"  ⚠️  Fallos          : {', '.join(failed_tickers)}")
    else:
        print(f"  Estado             : ✅ Sin errores")
    print("=" * 55)
    
    return total_inserted


if __name__ == "__main__":
    run_ingestion()