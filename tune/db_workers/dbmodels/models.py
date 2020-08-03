from datetime import datetime

__all__ = [
    "SqlTune",
    "SqlJob",
    "SqlUCIParam",
    "SqlTimeControl",
    "SqlResult",
]
try:
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
        Table,
    )
    from sqlalchemy.orm import relationship

    from tune.db_workers.dbmodels import Base
    from tune.db_workers.utils import TimeControl

    SCHEMA = "new"

    class SqlTune(Base):
        __tablename__ = "tunes"
        __table_args__ = {"schema": SCHEMA}
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
        weight = Column(Numeric, default=1.0, nullable=False)
        description = Column(String(250), nullable=True)

        jobs = relationship("SqlJob", back_populates="tune", cascade="all")

        def __repr__(self):
            return f"<Tune (id={self.id}, timestamp={self.timestamp}, weight={self.weight})>"

    job_tc_table = Table(
        "jobstotimes",
        Base.metadata,
        Column("job_id", Integer, ForeignKey(f"{SCHEMA}.jobs.id")),
        Column("tc_id", Integer, ForeignKey(f"{SCHEMA}.timecontrols.id")),
        schema=SCHEMA,
    )

    class SqlJob(Base):
        __tablename__ = "jobs"
        __table_args__ = {"schema": SCHEMA}
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
        active = Column(Boolean, default=True, nullable=False)
        weight = Column(Numeric, default=1.0, nullable=False)
        minimum_version = Column(Integer, default=1, nullable=False)
        maximum_version = Column(Integer, nullable=True, default=None)
        minimum_samplesize = Column(Integer, nullable=False, default=16)
        config = Column(JSON)
        engine1_exe = Column(String(100), nullable=False, default="lc0")
        engine1_nps = Column(Numeric, nullable=False)
        engine2_exe = Column(String(100), nullable=False, default="sf")
        engine2_nps = Column(Numeric, nullable=False)

        tune_id = Column(Integer, ForeignKey(f"{SCHEMA}.tunes.id"))
        tune = relationship("SqlTune", back_populates="jobs")
        params = relationship("SqlUCIParam", back_populates="job", cascade="all")
        time_controls = relationship("SqlTimeControl", secondary=job_tc_table)
        results = relationship("SqlResult", back_populates="job")

        def __repr__(self):
            return (
                f"<Job (id={self.id}, timestamp={self.timestamp}, active={self.active},"
                f" weight={self.weight}, engine1_exe={self.engine1_exe}, engine1_nps={self.engine1_nps},"
                f" engine2_exe={self.engine2_exe}, engine2_nps={self.engine2_nps}, config={self.config})>"
            )

    class SqlUCIParam(Base):
        __tablename__ = "params"
        __table_args__ = (
            PrimaryKeyConstraint("key", "job_id", name="param_pk"),
            {"schema": SCHEMA},
        )
        key = Column(String(100))
        value = Column(String(250), nullable=False)
        job_id = Column(Integer, ForeignKey(f"{SCHEMA}.jobs.id"))
        job = relationship("SqlJob", back_populates="params")

        def __repr__(self):
            return f"<UCI (key={self.key}, value={self.value}, job_id={self.job_id})>"

    class SqlTimeControl(Base):
        __tablename__ = "timecontrols"
        __table_args__ = ({"schema": SCHEMA},)
        id = Column(Integer, primary_key=True)
        engine1_time = Column(Numeric, nullable=False)
        engine1_increment = Column(Numeric, nullable=True, default=None)
        engine2_time = Column(Numeric, nullable=False)
        engine2_increment = Column(Numeric, nullable=True, default=None)
        draw_rate = Column(Numeric, nullable=False, default=0.33)

        def __repr__(self):
            engine1_inc = (
                "" if self.engine1_increment is None else f"+{self.engine1_increment}"
            )
            engine2_inc = (
                "" if self.engine2_increment is None else f"+{self.engine2_increment}"
            )
            return (
                f"<TC (id={self.id}, engine1={self.engine1_time}{engine1_inc},"
                f" engine2={self.engine2_time}{engine2_inc}, draw_rate={self.draw_rate})>"
            )

        def to_tuple(self):
            return TimeControl(
                engine1_time=self.engine1_time,
                engine1_increment=self.engine1_increment,
                engine2_time=self.engine2_time,
                engine2_increment=self.engine2_increment,
            )

    class SqlResult(Base):
        __tablename__ = "results"
        __table_args__ = ({"schema": SCHEMA},)
        job_id = Column(Integer, ForeignKey(f"{SCHEMA}.jobs.id"), primary_key=True)
        tc_id = Column(
            Integer, ForeignKey(f"{SCHEMA}.timecontrols.id"), primary_key=True
        )
        ww_count = Column(Integer, default=0)
        wl_count = Column(Integer, default=0)
        wd_count = Column(Integer, default=0)
        dd_count = Column(Integer, default=0)
        dl_count = Column(Integer, default=0)
        ll_count = Column(Integer, default=0)
        job = relationship("SqlJob", back_populates="results")
        time_control = relationship("SqlTimeControl")

        def __repr__(self):
            return (
                f"<Result (job_id={self.job_id}, tc_id={self.tc_id},"
                f" ww={self.ww_count}, wd={self.wd_count},"
                f" wl={self.wl_count}, dd={self.dd_count},"
                f" dl={self.dl_count}, ll={self.ll_count})>"
            )


except ImportError:
    SqlTune = None
    SqlJob = None
    SqlUCIParam = None
    SqlTimeControl = None
    SqlResult = None
