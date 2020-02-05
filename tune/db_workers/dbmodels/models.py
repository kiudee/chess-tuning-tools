from datetime import datetime
import time

from sqlalchemy import (
    Column,
    Boolean,
    Numeric,
    String,
    ForeignKey,
    Integer,
    DateTime,
    BigInteger,
    PrimaryKeyConstraint,
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
    engine1_nps = Column(Numeric, nullable=False)
    engine2_nps = Column(Numeric, nullable=False)

    tune_id = Column(Integer, ForeignKey("tunes.id"))
    tune = relationship("SqlTune", back_populates="jobs")
    params = relationship("SqlUCIParam", back_populates="job", cascade="all")


class SqlUCIParam(Base):
    __tablename__ = "params"
    __table_args__ = (
        PrimaryKeyConstraint("key", "job_id", name="param_pk"),
        {"schema": SCHEMA},
    )
    key = Column(String(100))
    value = Column(String(250), nullable=False)
    job_id = Column(Integer, ForeignKey="jobs.id")
    job = relationship("SqlJob", back_populates="params")


class SqlTimeControl(Base):
    __tablename__ = "timecontrols"
    __table_args__ = (
        PrimaryKeyConstraint(
            "engine1_time",
            "engine1_increment",
            "engine2_time",
            "engine2_increment",
            "job_id",
            name="timecontrol_pk",
        ),
        {"schema": SCHEMA},
    )
    engine1_time = Column(Numeric, nullable=False)
    engine1_increment = Column(Numeric, nullable=True, default=None)
    engine2_time = Column(Numeric, nullable=False)
    engine2_increment = Column(Numeric, nullable=True, default=None)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    job = relationship("SqlJob", back_populates="timecontrols")
    id = Column(Integer, autoincrement=True)


class SqlResult(Base):
    __tablename__ = "results"
    __table_args__ = ({"schema": SCHEMA},)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    tc_id = Column(Integer, ForeignKey("timecontrols.id"))
