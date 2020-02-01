"""The server worker, which reads results and schedules new jobs.

Usage:
  tuning_server.py (run | deactivate | reactivate) [options] <experiment_file> <dbconfig>
  tuning_server.py -h | --help

Options:
  -h --help         Show this screen.
  -v --verbose      Set log level to debug.
  --logfile=<path>  Path to where the log is output to

"""
import joblib
import json
import logging
import os
import psycopg2
import re
import sys
import numpy as np
import pytz
import scipy.stats
from datetime import datetime
from time import sleep
from psycopg2.extras import DictCursor

from bask import Optimizer
from skopt.space import space as skspace
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
from skopt.utils import normalize_dimensions

from .utils import parse_timecontrol, MatchResult, TimeControl
from ..io import InitStrings

__all__ = ["TuningServer"]


class TuningServer(object):
    def __init__(self, experiment_path, dbconfig_path, **kwargs):
        self.logger = logging.getLogger("TuningServer")
        self.experiment_path = experiment_path
        if os.path.isfile(dbconfig_path):
            with open(dbconfig_path, "r") as config_file:
                config = config_file.read().replace("\n", "")
                self.logger.debug(f"Reading DB config:\n{config}")
                self.connect_params = json.loads(config)
        else:
            raise ValueError("No dbconfig file found at provided path")

        if os.path.isfile(experiment_path):
            with open(experiment_path, "r+") as experiment_file:
                exp = experiment_file.read().replace("\n", "")
                self.logger.debug(f"Reading experiment config:\n{exp}")
                self.experiment = json.loads(exp)
                self.logger.debug(f"self.experiment = \n{self.experiment}")
        else:
            raise ValueError("No experiment config file found at provided path")
        self.rng = np.random.RandomState(self.experiment.get("random_seed", 123))
        self.setup_tuner()

        try:
            os.makedirs("experiments")
        except FileExistsError:
            pass
        # TODO: in principle after deleting all jobs from the database,
        #       this could be problematic:
        self.pos = None
        self.chain = None
        if "tune_id" in self.experiment:
            self.resume_tuning()

    def write_experiment_file(self):
        with open(self.experiment_path, "w") as experiment_file:
            experiment_file.write(json.dumps(self.experiment, indent=2))

    def save_state(self):
        path = os.path.join("experiments", f"data_tuneid_{self.experiment['tune_id']}.npz")
        np.savez_compressed(path, np.array(self.opt.gp.pos_), np.array(self.opt.gp.chain_))

    def resume_tuning(self):
        path = os.path.join("experiments", f"data_tuneid_{self.experiment['tune_id']}.npz")
        if os.path.exists(path):
            data = np.load(path)
            self.opt.gp.pos_ = data["arr_0"]
            self.opt.gp.chain_ = data["arr_1"]

    def parse_dimensions(self, param_dict):
        def floatify(s):
            try:
                return float(s)
            except ValueError:
                return s

        dimensions = []
        for s in param_dict.values():
            prior_str = re.findall(r"(\w+)\(", s)[0]
            prior_param_strings = re.findall(r"\((.*?)\)", s)[0].split(",")
            keys = [x.split("=")[0].strip() for x in prior_param_strings]
            vals = [floatify(x.split("=")[1].strip()) for x in prior_param_strings]
            dim = getattr(skspace, prior_str)(**dict(zip(keys, vals)))
            dimensions.append(dim)
        return dimensions

    def parse_priors(self, priors):
        if isinstance(priors, str):
            try:
                result = joblib.load(priors)
            except IOError:
                self.logger.error(f"Priors could not be loaded from path {priors}. Terminating...")
                sys.exit(1)
        else:
            result = []
            for i, p in enumerate(priors):
                prior_str = re.findall(r"(\w+)\(", p)[0]
                prior_param_strings = re.findall(r"\((.*?)\)", p)[0].split(",")
                keys = [x.split("=")[0].strip() for x in prior_param_strings]
                vals = [float(x.split("=")[1].strip()) for x in prior_param_strings]
                dist = getattr(scipy.stats, prior_str)(**dict(zip(keys, vals)))
                if i == 0 or i == len(priors) - 1:
                    # The signal variance and the signal noise are in positive, sqrt domain
                    prior = lambda x: dist.logpdf(np.sqrt(np.exp(x))) + x / 2.0 - np.log(2.0)
                else:
                    # The lengthscale(s) are in positive domain
                    prior = lambda x: dist.logpdf(np.exp(x)) + x
                result.append(prior)
        return result

    def setup_tuner(self):
        self.tunecfg = self.experiment["tuner"]
        self.dimensions = self.parse_dimensions(self.tunecfg["parameters"])
        self.space = normalize_dimensions(self.dimensions)
        self.priors = self.parse_priors(self.tunecfg["priors"])

        self.kernel = ConstantKernel(
            constant_value=self.tunecfg.get("variance_value", 0.1 ** 2),
            constant_value_bounds=tuple(self.tunecfg.get("variance_bounds", (0.01 ** 2, 0.5 ** 2))),
        ) * Matern(
            length_scale=self.tunecfg.get("length_scale_value", 0.3),
            length_scale_bounds=tuple(self.tunecfg.get("length_scale_bounds", (0.2, 0.8))),
            nu=2.5,
        )
        self.opt = Optimizer(
            dimensions=self.dimensions,
            n_points=self.tunecfg.get("n_points", 1000),
            n_initial_points=self.tunecfg.get("n_initial_points", 5 * len(self.dimensions)),
            gp_kernel=self.kernel,
            gp_kwargs=dict(normalize_y=True),
            gp_priors=self.priors,
            acq_func=self.tunecfg.get("acq_func", "ts"),
            acq_func_kwargs=self.tunecfg.get("acq_func_kwargs", None),  # TODO: Check if this works for all parameters
            random_state=self.rng.randint(0, np.iinfo(np.int32).max),
        )

    def query_data(self, cursor, tune_id, include_active=False):
        # TODO: These scores are linear in the W/L difference, maybe use something proper like (regularized) Elo
        # TODO: Support shrinking of ranges, by filtering X, y here
        if include_active:
            query = """
            select X, -avg((w-l) / (w + l - power(w-l, 2))) as y from
            (
                select X, (wins::decimal+0.5) / (wins + draws + losses + 1.5) as w, (losses::decimal+0.5) / (wins + draws + losses + 1.5) as l
                from tuning_jobs
                natural inner join tuning_results
                where tune_id = %(tune_id)s
            ) as xyz
            group by X;
            """
        else:
            query = """
            select X, -avg((w-l) / (w + l - power(w-l, 2))) as y from
            (
                select X, (wins::decimal+0.5) / (wins + draws + losses + 1.5) as w, (losses::decimal+0.5) / (wins + draws + losses + 1.5) as l
                from tuning_jobs
                natural inner join tuning_results
                where tune_id = %(tune_id)s and not active
            ) as xyz
            group by X;
            """
        cursor.execute(query, {"tune_id": tune_id})
        result = cursor.fetchall()
        X = np.array([x[0] for x in result], dtype=np.float64)
        y = np.array([x[1] for x in result], dtype=np.float64)
        self.logger.debug(f"Query yielded X, y =\n{X}\n{y}")

        query = """
        select wins+losses+draws as total
        from tuning_jobs natural inner join tuning_results
        where tune_id=%(tune_id)s and active;
        """
        cursor.execute(query, {"tune_id": tune_id})
        samplesize_reached = np.all(np.array(cursor.fetchall()) >= self.experiment.get("minimum_samplesize", 16))
        return X, y, samplesize_reached

    @staticmethod
    def change_engine_config(engine_config, params):
        init_strings = InitStrings(engine_config[0]["initStrings"])  # TODO: allow tuning of different index
        for k, v in params.items():
            init_strings[k] = v

    def insert_jobs(self, conn, cursor, new_x):
        # 2. First set all active jobs to inactive:
        try:
            cursor.execute(
                """
            update tuning_jobs set active=false where tune_id=%(tune_id)s;
            """,
                {"tune_id": self.experiment["tune_id"]},
            )

            # 3. Insert new jobs:
            job_dict = {"engine": self.experiment["engine"], "cutechess": self.experiment["cutechess"]}
            timestamp = datetime.utcnow().replace(tzinfo=pytz.utc)
            for i, tc in enumerate(self.experiment["time_controls"]):
                job_dict["time_control"] = tc
                job_json = json.dumps(job_dict)

                query = """
                insert into tuning_jobs
                    (timestamp, config, active, tune_id, job_weight, minimum_version, lc0_nodes, sf_nodes, x)
                    values
                    (%(timestamp)s, %(config)s, %(active)s, %(tune_id)s, %(job_weight)s, %(minimum_version)s,
                    %(lc0_nodes)s, %(sf_nodes)s, %(new_x)s)
                    returning job_id;
                """
                cursor.execute(
                    query,
                    {
                        "timestamp": timestamp,
                        "config": job_json,
                        "active": True,
                        "tune_id": self.experiment["tune_id"],
                        "job_weight": self.experiment.get("job_weight", 1.0),
                        "minimum_version": self.experiment.get("minimum_version", 1),
                        "lc0_nodes": self.experiment["lc0_nodes"],
                        "sf_nodes": self.experiment["sf_nodes"],
                        "new_x": new_x,
                    },
                )
                job_id = cursor.fetchone()[0]

                query = """
                insert into tuning_results
                (job_id, tune_id, time_control, wins, losses, draws)
                values
                (%(job_id)s, %(tune_id)s, %(time_control)s, 0, 0, 0);
                """
                cursor.execute(
                    query, {"job_id": job_id, "tune_id": self.experiment["tune_id"], "time_control": str(tc)}
                )
            conn.commit()
        except BaseException:
            conn.rollback()

    def run(self):
        # 0. Before we run the main loop, do we need to initialize or resume?
        #    * Resume from files (in experiment folder)
        #    * Create tune entry in db if it does not exist yet

        with psycopg2.connect(**self.connect_params) as conn:
            with conn.cursor(cursor_factory=DictCursor) as curs:
                if "tune_id" not in self.experiment:
                    # This appears to be a new tune, create entry in tunes database:
                    curs.execute(
                        """
                        insert into tunes (description) VALUES (%(desc)s) returning tune_id;
                        """,
                        {"desc": self.experiment.get("description", "This job does not have a description")},
                    )
                    self.experiment["tune_id"] = curs.fetchone()[0]
                    self.write_experiment_file()
        while True:
            # 1. Check if there are currently running jobs
            # TODO: Case when there are no jobs yet!
            # 2. Check if minimum sample size and minimum wait time are reached, then query data and update model:
            with psycopg2.connect(**self.connect_params) as conn:
                with conn.cursor(cursor_factory=DictCursor) as curs:
                    X, y, samplesize_reached = self.query_data(curs, self.experiment["tune_id"], include_active=True)
                    self.logger.debug(f"Queried the database for data and got (last 5):\n{X[-5:]}\n{y[-5:]}")
                    if len(X) == 0:
                        self.logger.info("There are no datapoints yet, start first job")
                        new_x = self.opt.ask()
                        # Alter engine json using Initstrings
                        params = dict(zip(self.experiment["tuner"]["parameters"].keys(), new_x))
                        self.change_engine_config(self.experiment["engine"], params)
                        with psycopg2.connect(**self.connect_params) as conn:
                            with conn.cursor(cursor_factory=DictCursor) as curs:
                                self.insert_jobs(conn=conn, cursor=curs, new_x=new_x)
                        self.logger.info("New jobs committed to database.")
                        samplesize_reached = False

            if not samplesize_reached:
                sleep_seconds = self.experiment.get("sleep_time")
                self.logger.debug(f"Required sample size not yet reached. Sleeping {sleep_seconds} seconds.")
                sleep(sleep_seconds)
                continue

            now = datetime.now()
            self.opt.tell(
                X.tolist(),
                y.tolist(),
                fit=True,
                replace=True,
                n_samples=self.tunecfg["n_samples"],
                gp_samples=self.tunecfg["gp_samples"],
                gp_burnin=self.tunecfg["gp_burnin"],
                progress=False,
            )
            later = datetime.now()
            difference = (later - now).total_seconds()
            self.logger.info(f"Calculating GP posterior and acquisition function finished in {difference}s")
            self.logger.info(f"Current GP kernel:\n{self.opt.gp.kernel_}")
            if self.opt.gp.chain_ is not None:
                self.logger.debug("Saving position and chain")
                self.save_state()

            new_x = self.opt.ask()
            # Alter engine json using Initstrings
            params = dict(zip(self.experiment["tuner"]["parameters"].keys(), new_x))
            self.change_engine_config(self.experiment["engine"], params)
            with psycopg2.connect(**self.connect_params) as conn:
                with conn.cursor(cursor_factory=DictCursor) as curs:
                    self.insert_jobs(conn=conn, cursor=curs, new_x=new_x)
            self.logger.info("New jobs committed to database.")

    def deactivate(self):
        raise NotImplementedError

    def reactivate(self):
        raise NotImplementedError
