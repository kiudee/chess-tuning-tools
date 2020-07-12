"""Console script for chess_tuning_tools."""
from datetime import datetime
import json
import logging
import pathlib
import sys

from bask.optimizer import Optimizer
import click
import matplotlib.pyplot as plt
import numpy as np


from tune.db_workers import TuningServer, TuningClient
from tune.local import run_match, parse_experiment_result
from tune.io import prepare_engines_json, load_tuning_config, write_engines_json


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Turn on debug output."
)
@click.option("--logfile", default=None, help="Path to where the log is saved to.")
@click.option(
    "--terminate-after", default=0, help="Terminate the client after x minutes."
)
@click.option(
    "--clientconfig", default=None, help="Path to the client configuration file."
)
@click.argument("dbconfig")
def run_client(verbose, logfile, terminate_after, clientconfig, dbconfig):
    """ Run the client to generate games for distributed tuning.

    In order to connect to the database you need to provide a valid DBCONFIG
    json file. It contains the necessary parameters to connect to the database
    where it can fetch jobs and store results.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        filename=logfile,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    tc = TuningClient(
        dbconfig_path=dbconfig,
        terminate_after=terminate_after,
        clientconfig=clientconfig,
    )
    tc.run()


@cli.command()
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Turn on debug output."
)
@click.option("--logfile", default=None, help="Path to where the log is saved to.")
@click.argument("command")
@click.argument("experiment_file")
@click.argument("dbconfig")
def run_server(verbose, logfile, command, experiment_file, dbconfig):
    """Run the tuning server for a given EXPERIMENT_FILE (json).

    To connect to the database you also need to provide a DBCONFIG json file.

    \b
    You can choose from these COMMANDs:
     * run: Starts the server.
     * deactivate: Deactivates all active jobs of the given experiment.
     * reactivate: Reactivates all recent jobs for which sample size is not reached yet.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        filename=logfile,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    tc = TuningServer(experiment_path=experiment_file, dbconfig_path=dbconfig)
    if command == "run":
        tc.run()
    elif command == "deactivate":
        tc.deactivate()
    elif command == "reactivate":
        tc.reactivate()
    else:
        raise ValueError(f"Command {command} is not recognized. Terminating...")


@cli.command()
@click.option(
    "-c",
    "--tuning-config",
    help="json file containing the tuning configuration.",
    required=True,
    type=str,
)
@click.option(
    "--data-path", default=None, help="Save the evaluated points to this file."
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Let the optimizer resume, if it finds points it can use.",
)
@click.option(
    "--n-points",
    default=500,
    help="The number of random points to consider as the next point.",
)
@click.option(
    "--n-initial-points", default=30, help="Size of initial dense set of points to try."
)
@click.option(
    "--random-seed",
    default=0,
    help="Number to seed all internally used random generators.",
)
@click.option("--logfile", default="log.txt", help="Path to log file.")
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Turn on debug output."
)
def local(
    tuning_config,
    data_path=None,
    resume=True,
    n_points=500,
    n_initial_points=30,
    random_seed=0,
    logfile="log.txt",
    verbose=False,
):
    """Run a local tune.

    Parameters defined in the `tuning_config` file always take precedence.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    #logging.basicConfig(
    #    level=log_level,
    #    filename=logfile,
    #    format=,
    #    datefmt="%Y-%m-%d %H:%M:%S",
    #)
    file_logger = logging.FileHandler(logfile)
    file_logger.setFormatter(log_format)
    root_logger.addHandler(file_logger)
    console_logger = logging.StreamHandler(sys.stdout)
    console_logger.setFormatter(log_format)
    root_logger.addHandler(console_logger)
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Read tuning configuration:
    conf_path = pathlib.Path(tuning_config)
    if conf_path.exists():
        with open(conf_path) as json_file:
            json_dict = json.load(json_file)
            logging.debug(f"Got the following tuning settings:\n{json_dict}")
            settings, commands, fixed_params, param_ranges = load_tuning_config(
                json_dict
            )
    else:
        raise ValueError(f"No tuning configuration file found here: {conf_path}")

    # 1. Create seed sequence
    ss = np.random.SeedSequence(random_seed)
    # 2. Create kernel
    # 3. Create optimizer
    random_state = np.random.RandomState(np.random.MT19937(ss.spawn(1)[0]))
    opt = Optimizer(
        dimensions=list(param_ranges.values()),
        n_points=n_points,
        n_initial_points=n_initial_points,
        # gp_kernel=kernel,  # TODO: Let user pass in different kernels
        gp_kwargs=dict(normalize_y=True),
        # gp_priors=priors,  # TODO: Let user pass in priors
        acq_func="pvrs",
        acq_func_kwargs=dict(alpha="inf", n_thompson=20),
        random_state=random_state,
    )
    X = []
    y = []
    noise = []
    iteration = 0

    # 3.1 Resume from existing data:
    if data_path is None:
        data_path = "data.npz"
    if resume:
        path = pathlib.Path(data_path)
        if path.exists():
            importa = np.load(path)
            X = importa["arr_0"].tolist()
            y = importa["arr_1"].tolist()
            noise = importa["arr_2"].tolist()
            iteration = len(X)
            logging.info(f"Importing {iteration} existing datapoints. This could take a while...")
            opt.tell(
                X,
                y,
                noise_vector=noise,
                gp_burnin=300,
                gp_samples=100,
                n_samples=0,
                progress=True,
            )
            logging.info("Importing finished.")

    # 4. Main optimization loop:
    while True:
        logging.info("Starting iteration {}".format(iteration))
        point = opt.ask()
        point_dict = dict(zip(param_ranges.keys(), point))
        logging.info("Testing {}".format(point_dict))

        engine_json = prepare_engines_json(commands=commands, fixed_params=fixed_params)
        logging.debug(f"engines.json is prepared:\n{engine_json}")
        write_engines_json(engine_json, point_dict)
        logging.info("Start experiment")
        now = datetime.now()
        out_exp, out_exp_err = run_match(**settings)
        later = datetime.now()
        difference = (later - now).total_seconds()
        logging.info(f"Experiment finished ({difference}s elapsed)")
        logging.debug(f"Raw result:\n{out_exp}\n{out_exp_err}")

        score, error = parse_experiment_result(out_exp, **settings)
        logging.info("Got score: {} +- {}".format(score, error))
        logging.info("Updating model")
        while True:
            try:
                now = datetime.now()
                # We fetch kwargs manually here to avoid collisions:
                n_samples = settings.get("n_samples", 0)
                gp_burnin = settings.get("gp_burnin", 5)
                gp_samples = settings.get("gp_samples", 200)
                if opt.gp.chain_ is None:
                    gp_burnin = settings.get("gp_initial_burnin", 200)
                    gp_samples = settings.get("gp_initial_samples", 300)
                    opt.tell(
                        point,
                        score,
                        n_samples=n_samples,
                        gp_samples=gp_samples,
                        gp_burnin=gp_burnin,
                    )
                else:
                    opt.tell(
                        point,
                        score,
                        n_samples=n_samples,
                        gp_samples=gp_samples,
                        gp_burnin=gp_burnin,
                    )
                later = datetime.now()
                difference = (later - now).total_seconds()
                logging.info(f"GP sampling finished ({difference}s)")
                logging.debug(f"GP kernel: {opt.gp.kernel_}")
            except ValueError:
                logging.warning("Error encountered during fitting."
                                "Trying to sample chain a bit.")
                opt.gp.sample(n_burnin=5, priors=opt.gp_priors)
            else:
                break
        X.append(point)
        y.append(score)
        noise.append(error)
        iteration = len(X)
        np.savez_compressed(data_path, np.array(X), np.array(y), np.array(noise))


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
