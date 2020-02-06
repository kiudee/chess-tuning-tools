from datetime import datetime
import time

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    PrimaryKeyConstraint,
    String,
)
from sqlalchemy.orm import relationship

from tune.db_workers.dbmodels import Base

SCHEMA = "new"


class SqlTune(Base):
    __tablename__ = "tunes"
    __table_args__ = {"schema": SCHEMA}
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    weight = Column(Numeric, default=1.0, nullable=False)

    jobs = relationship("SqlJob", back_populates="tune", cascade="all")


class SqlJob(Base):
    __tablename__ = "jobs"
    __table_args__ = {"schema": SCHEMA}
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    weight = Column(Numeric, default=1.0, nullable=False)
    minimum_version = Column(Integer, default=1, nullable=False)
    maximum_version = Column(Integer, nullable=True, default=None)
    config = Column(JSON)
    engine1_exe = Column(String(100), nullable=False, default="lc0")
    engine1_nps = Column(Numeric, nullable=False)
    engine2_exe = Column(String(100), nullable=False, default="sf")
    engine2_nps = Column(Numeric, nullable=False)

    tune_id = Column(Integer, ForeignKey("new.tunes.id"))
    tune = relationship("SqlTune", back_populates="jobs")
    params = relationship("SqlUCIParam", back_populates="job", cascade="all")
    time_controls = relationship("SqlTimeControlMatch")


class SqlUCIParam(Base):
    __tablename__ = "params"
    __table_args__ = (
        PrimaryKeyConstraint("key", "job_id", name="param_pk"),
        {"schema": SCHEMA},
    )
    key = Column(String(100))
    value = Column(String(250), nullable=False)
    job_id = Column(Integer, ForeignKey("new.jobs.id"))
    job = relationship("SqlJob", back_populates="params")


class SqlTimeControl(Base):
    __tablename__ = "timecontrols"
    __table_args__ = (
        {"schema": SCHEMA},
    )
    id = Column(Integer, primary_key=True)
    engine1_time = Column(Numeric, nullable=False)
    engine1_increment = Column(Numeric, nullable=True, default=None)
    engine2_time = Column(Numeric, nullable=False)
    engine2_increment = Column(Numeric, nullable=True, default=None)
    draw_rate = Column(Numeric, nullable=False, default=0.33)


class SqlTimeControlMatch(Base):
    __tablename__ = "jobstotimes"
    __table_args__ = (
        {"schema": SCHEMA},
    )
    job_id = Column(Integer, ForeignKey("new.jobs.id"), primary_key=True)
    tc_id = Column(Integer, ForeignKey("new.timecontrols.id"), primary_key=True)
    times = relationship("SqlTimeControl")


class SqlResult(Base):
    __tablename__ = "results"
    __table_args__ = ({"schema": SCHEMA},)
    job_id = Column(Integer, ForeignKey("new.jobs.id"), primary_key=True)
    tc_id = Column(Integer, ForeignKey("new.timecontrols.id"), primary_key=True)
    ww_count = Column(Integer, default=0)
    wl_count = Column(Integer, default=0)
    wd_count = Column(Integer, default=0)
    dd_count = Column(Integer, default=0)
    dl_count = Column(Integer, default=0)
    ll_count = Column(Integer, default=0)
