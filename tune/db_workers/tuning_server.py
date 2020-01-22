"""The server worker, which reads results and schedules new jobs.

Usage:
  tuning_server.py (run | deactivate | reactivate) [options] <experiment_file> <dbconfig>
  tuning_server.py -h | --help

Options:
  -h --help         Show this screen.
  -v --verbose      Set log level to debug.
  --logfile=<path>  Path to where the log is output to

"""
from docopt import docopt
import json
import logging
import os
import psycopg2
import re
import subprocess
import sys
import numpy as np
from time import sleep
from psycopg2.extras import DictCursor

from bask import Optimizer
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
from skopt.utils import normalize_dimensions

from .utils import parse_timecontrol, MatchResult, TimeControl


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
                self.experiment = json.loads(config)
        else:
            raise ValueError("No experiment config file found at provided path")
        self.rng = np.random.RandomState(self.experiment.get('random_seed', 123))
        self.setup_tuner()

        try:
            os.makedirs('experiments')
        except FileExistsError:
            pass
        # TODO: in principle after deleting all jobs from the database,
        #       this could be problematic:
        if 'tune_id' in self.experiment:
            self.resume_tuning()

    def resume_tuning(self):
        # 1. Load appropriate files:
        #    * Chain
        #    * Position
        # 2. Read tune_id and query corresponding active jobs
        # Either:
        # 3a. Run quick init without sampling (setting position, chain and theta)
        # 3b. Avoid initialization at all and defer to run (to avoid slowdown in deactivate/reactivate)
        raise NotImplementedError

    def write_experiment_file(self):
        with open(self.experiment_path, "w") as experiment_file:
            experiment_file.write(json.dumps(self.experiment))

    def parse_dimensions(self, param_dict):
        raise NotImplementedError

    def parse_priors(self, priors):
        raise NotImplementedError

    def setup_tuner(self):
        tunecfg = self.experiment['tuner']
        self.dimensions = self.parse_dimensions(tunecfg['parameters'])
        self.space = normalize_dimensions(self.dimensions)
        self.priors = self.parse_priors(tunecfg['priors'])

        self.kernel = (
            ConstantKernel(constant_value=tunecfg.get("variance_value", 0.1 ** 2),
                           constant_value_bounds=tuple(tunecfg.get("variance_bounds", (0.01 ** 2, 0.5 ** 2))))
            *  Matern(
                length_scale=tunecfg.get("length_scale_value", 0.3),
                length_scale_bounds=tuple(tunecfg.get("length_scale_bounds", (0.2, 0.8))),
                nu=2.5
            )
        )
        self.opt = Optimizer(
            dimensions=self.dimensions,
            n_points=tunecfg.get('n_points', 1000),
            n_initial_points=tunecfg.get('n_initial_points', 5 * len(self.dimensions)),
            gp_kernel=self.kernel,
            gp_kwargs=dict(normalize_y=True),
            gp_priors=self.priors,
            acq_func=tunecfg.get('acq_func', 'ts'),
            acq_func_kwargs=tunecfg.get('acq_func_kwargs', None),  # TODO: Check if this works for all parameters
            random_state=self.rng.randint(0, np.iinfo(np.int32).max)
        )


    def run(self):
        while True:
            # Functionality:
            # * Parse config
            # * Instantiate tuner with ranges and priors
            # * Resume tuning from something (progress file(s)?)
            # * Commit new jobs to tuning_jobs and tuning_results
            # * Pool results in regular intervals and update tuner
            #   - Able to revise older datapoints?
            # *

    def deactivate(self):
        raise NotImplementedError

    def reactivate(self):
        raise NotImplementedError


if __name__ == "__main__":
    arguments = docopt(__doc__)
    log_level = logging.DEBUG if arguments['--verbose'] else logging.INFO
    logging.basicConfig(level=log_level, filename=arguments['--logfile'])
    tc = TuningServer(
        experiment_path=arguments['<experiment_file>'],
        dbconfig_path=arguments['<dbconfig>']
    )
    if arguments['run']:
        tc.run()
    elif arguments['deactivate']:
        tc.deactivate()
    elif arguments['reactivate']:
        tc.reactivate()
