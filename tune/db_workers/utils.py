from collections import namedtuple
from contextlib import contextmanager
from sqlalchemy import create_engine

MatchResult = namedtuple("MatchResult", ["wins", "losses", "draws"])
TimeControl = namedtuple("TimeControl", ["engine1", "engine2"])


def parse_timecontrol(tc_string):
    if "+" in tc_string:
        return tuple([float(x) for x in tc_string.split("+")])
    return (float(tc_string),)


def get_session_maker(sessionmaker):
    @contextmanager
    def make_session():
        session = sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return make_session


def create_sqlalchemy_engine(config):
    db_uri = (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['dbname']}"
    )
    return create_engine(db_uri, pool_pre_ping=True)
