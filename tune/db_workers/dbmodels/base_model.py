__all__ = ["Base"]

try:
    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()
except ImportError:
    Base = None
    declarative_base = None
