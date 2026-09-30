from sqlalchemy.engine import Engine
from backend.database.connection import Base, engine
import backend.database.models  # Ensure all models are registered on Base.metadata


def init_db(target_engine: Engine = None) -> None:
    """Creates all database tables."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)
    print("Database tables initialized successfully.")


def drop_db(target_engine: Engine = None) -> None:
    """Drops all database tables."""
    eng = target_engine or engine
    Base.metadata.drop_all(bind=eng)
    print("Database tables dropped successfully.")


if __name__ == "__main__":
    init_db()
