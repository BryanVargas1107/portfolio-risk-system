-- ============================================================
-- PORTFOLIO RISK SYSTEM — Database Schema
-- ============================================================
-- Contexto de negocio: Este schema replica la estructura de
-- datos que usa un equipo de Risk Management para monitorear
-- diariamente la exposición al riesgo de mercado de un
-- portafolio de inversión.
-- ============================================================

-- ------------------------------------------------------------
-- TABLA 1: assets
-- Catálogo de activos financieros monitoreados.
-- En un banco real, esta tabla la mantiene el equipo de
-- Middle Office y es la fuente de verdad de los instrumentos
-- autorizados para trading.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    asset_id        SERIAL PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL UNIQUE,
    company_name    VARCHAR(100) NOT NULL,
    sector          VARCHAR(50),
    asset_type      VARCHAR(20)  NOT NULL, -- 'stock', 'etf', 'index'
    is_benchmark    BOOLEAN      DEFAULT FALSE,
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- TABLA 2: prices_daily
-- Precios históricos diarios de cada activo.
-- Equivale al "market data store" de cualquier institución
-- financiera — la fuente base para todos los cálculos de
-- riesgo y rendimiento.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices_daily (
    price_id        SERIAL PRIMARY KEY,
    asset_id        INT          NOT NULL REFERENCES assets(asset_id),
    price_date      DATE         NOT NULL,
    open_price      NUMERIC(12,4),
    high_price      NUMERIC(12,4),
    low_price       NUMERIC(12,4),
    close_price     NUMERIC(12,4) NOT NULL,
    adj_close       NUMERIC(12,4),
    volume          BIGINT,
    daily_return    NUMERIC(10,6), -- retorno logarítmico diario
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset_id, price_date)
);

-- ------------------------------------------------------------
-- TABLA 3: portfolio_metrics
-- Métricas de rendimiento calculadas diariamente.
-- Estas son las cifras que un Portfolio Manager revisa cada
-- mañana antes de que abra el mercado. Sharpe, Drawdown,
-- Beta y Alpha son métricas regulatorias y de gestión.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_metrics (
    metric_id           SERIAL PRIMARY KEY,
    calculation_date    DATE         NOT NULL UNIQUE,
    portfolio_value     NUMERIC(15,2),
    daily_return        NUMERIC(10,6),
    cumulative_return   NUMERIC(10,6),
    sharpe_ratio        NUMERIC(8,4),  -- rentabilidad ajustada al riesgo
    sortino_ratio       NUMERIC(8,4),  -- como Sharpe pero solo volatilidad negativa
    max_drawdown        NUMERIC(8,4),  -- mayor caída desde el máximo histórico
    beta                NUMERIC(8,4),  -- sensibilidad al movimiento del mercado
    alpha               NUMERIC(8,4),  -- rentabilidad sobre el benchmark
    volatility_30d      NUMERIC(8,4),  -- volatilidad anualizada 30 días
    created_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- TABLA 4: var_history
-- Historial de cálculos de Value at Risk (VaR).
-- El VaR es el cálculo de riesgo más importante en banca.
-- Basilea III exige que los bancos lo calculen y reporten
-- diariamente para determinar el capital regulatorio mínimo.
-- Guardamos los 3 métodos estándar de la industria.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS var_history (
    var_id              SERIAL PRIMARY KEY,
    calculation_date    DATE         NOT NULL UNIQUE,
    portfolio_value     NUMERIC(15,2) NOT NULL,
    
    -- Método 1: Histórico (usa distribución real de retornos)
    var_historical_95   NUMERIC(10,6),
    var_historical_99   NUMERIC(10,6),
    
    -- Método 2: Paramétrico (asume distribución normal)
    var_parametric_95   NUMERIC(10,6),
    var_parametric_99   NUMERIC(10,6),
    
    -- Método 3: Monte Carlo (simulación de 10,000 escenarios)
    var_montecarlo_95   NUMERIC(10,6),
    var_montecarlo_99   NUMERIC(10,6),
    
    -- Pérdida en valor absoluto (USD) para el reporte ejecutivo
    var_amount_usd      NUMERIC(15,2),
    
    -- Flag que activa la alerta por email
    threshold_breached  BOOLEAN      DEFAULT FALSE,
    
    created_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- TABLA 5: forecasts
-- Proyecciones de precio generadas por el modelo Prophet.
-- En la industria esto forma parte del proceso de
-- "stress testing" y planificación de escenarios que los
-- equipos de riesgo presentan en los comités de inversión.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id         SERIAL PRIMARY KEY,
    asset_id            INT          NOT NULL REFERENCES assets(asset_id),
    generated_date      DATE         NOT NULL, -- fecha en que se generó la predicción
    forecast_date       DATE         NOT NULL, -- fecha que se está prediciendo
    predicted_price     NUMERIC(12,4) NOT NULL,
    lower_bound         NUMERIC(12,4), -- intervalo de confianza inferior (80%)
    upper_bound         NUMERIC(12,4), -- intervalo de confianza superior (80%)
    model_version       VARCHAR(20)  DEFAULT 'prophet_v1',
    created_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset_id, generated_date, forecast_date)
);

-- ------------------------------------------------------------
-- TABLA 6: risk_alerts
-- Registro de todas las alertas de riesgo generadas.
-- Equivale al "alert log" que los sistemas de risk management
-- guardan para auditoría regulatoria. En un banco, este
-- historial debe conservarse por ley mínimo 5 años.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_alerts (
    alert_id            SERIAL PRIMARY KEY,
    alert_date          TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    alert_type          VARCHAR(50)  NOT NULL, -- 'VAR_BREACH', 'HIGH_VOLATILITY', etc.
    severity            VARCHAR(10)  NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    metric_name         VARCHAR(50),
    metric_value        NUMERIC(10,6),
    threshold_value     NUMERIC(10,6),
    message             TEXT,
    email_sent          BOOLEAN      DEFAULT FALSE,
    email_sent_at       TIMESTAMP
);

-- ------------------------------------------------------------
-- ÍNDICES — Optimización de consultas frecuentes
-- En producción, las consultas de riesgo corren cada mañana
-- sobre millones de registros. Los índices son críticos.
-- ------------------------------------------------------------
CREATE INDEX idx_prices_date     ON prices_daily(price_date);
CREATE INDEX idx_prices_asset    ON prices_daily(asset_id);
CREATE INDEX idx_var_date        ON var_history(calculation_date);
CREATE INDEX idx_forecast_asset  ON forecasts(asset_id);
CREATE INDEX idx_alerts_date     ON risk_alerts(alert_date);

-- ------------------------------------------------------------
-- DATOS INICIALES — Portafolio base del sistema
-- 5 acciones del S&P 500 de distintos sectores + benchmark.
-- Diversificación sectorial real: así lo haría un gestor.
-- ------------------------------------------------------------
INSERT INTO assets (ticker, company_name, sector, asset_type, is_benchmark) VALUES
    ('SPY',  'SPDR S&P 500 ETF',          'Index',          'etf',   TRUE),
    ('AAPL', 'Apple Inc.',                 'Technology',     'stock', FALSE),
    ('JPM',  'JPMorgan Chase & Co.',       'Financials',     'stock', FALSE),
    ('JNJ',  'Johnson & Johnson',          'Healthcare',     'stock', FALSE),
    ('XOM',  'Exxon Mobil Corporation',    'Energy',         'stock', FALSE),
    ('MSFT', 'Microsoft Corporation',      'Technology',     'stock', FALSE);