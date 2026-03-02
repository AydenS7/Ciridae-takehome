from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .settings import settings

class Base(DeclarativeBase):
    pass

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def init_db() -> None:
    # Ensure models are imported so Base.metadata includes them
    from . import models           # Run
    from . import models_items     # LineItem
    from . import models_roommap   # RoomMap
    from . import models_matches   # Match
    Base.metadata.create_all(bind=engine)
