# ============================================================
# database.py — Módulo de conexión a PostgreSQL
# ============================================================
# Contexto de negocio: En un sistema de riesgo real, la capa
# de conexión a base de datos está centralizada y nunca se
# duplica. Cualquier cambio de credenciales o host se hace
# en un solo lugar. Este módulo implementa ese patrón.
# ============================================================

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    """
    Crea y retorna el engine de conexión a PostgreSQL.
    Usa variables de entorno — las credenciales nunca
    están hardcodeadas en el código fuente.
    """
    db_host     = os.getenv("DB_HOST", "localhost")
    db_port     = os.getenv("DB_PORT", "5432")
    db_name     = os.getenv("DB_NAME", "portfolio_risk")
    db_user     = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")

    connection_string = (
        f"postgresql+psycopg2://{db_user}:{db_password}"
        f"@{db_host}:{db_port}/{db_name}"
    )

    engine = create_engine(
        connection_string,
        pool_size=5,          # conexiones simultáneas máximas
        pool_pre_ping=True,   # verifica conexión antes de usarla
        echo=False            # True para ver SQL en consola (debug)
    )
    return engine


def get_session():
    """
    Retorna una sesión de base de datos lista para usar.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def test_connection():
    """
    Verifica que la conexión a PostgreSQL funciona correctamente.
    Útil para diagnóstico y en el arranque del pipeline.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Conexión exitosa a PostgreSQL")
            print(f"   Versión: {version[:50]}")
            return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False


if __name__ == "__main__":
    test_connection()