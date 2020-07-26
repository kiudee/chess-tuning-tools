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
from skopt.utils import create_result


from tune.db_workers import TuningServer, TuningClient
from tune.local import run_match, parse_experiment_result
from tune.io import prepare_engines_json, load_tuning_config, write_engines_json
from tune.utils import expected_ucb


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
    "-a",
    "--acq-function",
    default="mes",
    help="Acquisition function to use for selecting points to try. "
    "Can be {mes, pvrs, ei, ts, vr}.",
)
@click.option(
    "--acq-function-samples",
    default=1,
    help="How many GP samples to average the acquisition function over. "
    "More samples will slow down the computation time, but might give more "
    "stable tuning results. Less samples on the other hand cause more exploration "
    "which could help avoid the tuning to get stuck.",
)
@click.option(
    "-c",
    "--tuning-config",
    help="json file containing the tuning configuration.",
    required=True,
    type=click.File("r"),
)
@click.option(
    "-d",
    "--data-path",
    default="data.npz",
    help="Save the evaluated points to this file.",
    type=click.Path(exists=False),
)
@click.option(
    "--gp-burnin",
    default=5,
    type=int,
    help="Number of samples to discard before sampling model parameters. "
    "This is used during tuning and few samples suffice.",
)
@click.option(
    "--gp-samples",
    default=300,
    type=int,
    help="Number of model parameters to sample for the model. "
    "This is used during tuning and it should be a multiple of 100.",
)
@click.option(
    "--gp-initial-burnin",
    default=100,
    type=int,
    help="Number of samples to discard before starting to sample the initial model "
    "parameters. This is only used when resuming or for the first model.",
)
@click.option(
    "--gp-initial-samples",
    default=300,
    type=int,
    help="Number of model parameters to sample for the initial model. "
    "This is only used when resuming or for the first model. "
    "Should be a multiple of 100.",
)
@click.option(
    "-l",
    "--logfile",
    default="log.txt",
    help="Path to log file.",
    type=click.Path(exists=False),
)
@click.option(
    "--n-initial-points", default=30, help="Size of initial dense set of points to try."
)
@click.option(
    "--n-points",
    default=500,
    help="The number of random points to consider as possible next point. "
    "Less points reduce the computation time of the tuner, but reduce "
    "the coverage of the space.",
)
@click.option(
    "--random-seed",
    default=0,
    help="Number to seed all internally used random generators.",
)
@click.option(
    "--result-every",
    default=5,
    help="Output the actual current optimum every n-th iteration."
    "The further you are in the tuning process, the longer this will take to "
    "compute. Consider increasing this number, if you do not need the output "
    "that often.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Let the optimizer resume, if it finds points it can use.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Turn on debug output."
)
def local(
    tuning_config,
    acq_function="mes",
    acq_function_samples=1,
    data_path=None,
    gp_burnin=5,
    gp_samples=300,
    gp_initial_burnin=100,
    gp_initial_samples=300,
    logfile="log.txt",
    n_initial_points=30,
    n_points=500,
    random_seed=0,
    result_every=5,
    resume=True,
    verbose=False,
):
    """Run a local tune.

    Parameters defined in the `tuning_config` file always take precedence.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    file_logger = logging.FileHandler(logfile)
    file_logger.setFormatter(log_format)
    root_logger.addHandler(file_logger)
    console_logger = logging.StreamHandler(sys.stdout)
    console_logger.setFormatter(log_format)
    root_logger.addHandler(console_logger)

    json_dict = json.load(tuning_config)
    logging.debug(f"Got the following tuning settings:\n{json_dict}")
    settings, commands, fixed_params, param_ranges = load_tuning_config(json_dict)

    # 1. Create seed sequence
    ss = np.random.SeedSequence(random_seed)
    # 2. Create kernel
    # 3. Create optimizer
    random_state = np.random.RandomState(np.random.MT19937(ss.spawn(1)[0]))
    opt = Optimizer(
        dimensions=list(param_ranges.values()),
        n_points=settings.get("n_points", n_points),
        n_initial_points=settings.get("n_initial_points", n_initial_points),
        # gp_kernel=kernel,  # TODO: Let user pass in different kernels
        gp_kwargs=dict(normalize_y=True),
        # gp_priors=priors,  # TODO: Let user pass in priors
        acq_func=settings.get("acq_function", acq_function),
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
            logging.info(
                f"Importing {iteration} existing datapoints. This could take a while..."
            )
            opt.tell(
                X,
                y,
                noise_vector=noise,
                gp_burnin=settings.get("gp_initial_burnin", gp_initial_burnin),
                gp_samples=settings.get("gp_initial_samples", gp_initial_samples),
                n_samples=settings.get("n_samples", 1),
                progress=True,
            )
            logging.info("Importing finished.")

    # 4. Main optimization loop:
    while True:
        logging.info("Starting iteration {}".format(iteration))
        if iteration % result_every == 0 and opt.gp.chain_ is not None:
            result_object = create_result(Xi=X, yi=y, space=opt.space, models=[opt.gp])
            try:
                best_point, best_value = expected_ucb(result_object, alpha=0.0)
                best_point_dict = dict(zip(param_ranges.keys(), best_point))
                logging.info(f"Current optimum:\n{best_point_dict}")
                logging.info(f"Estimated value: {best_value}")
            except ValueError:
                logging.info(
                    "Computing current optimum was not successful. "
                    "This can happen in rare cases and running the "
                    "tuner again usually works."
                )
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
                n_samples = settings.get("acq_function_samples", acq_function_samples)
                gp_burnin = settings.get("gp_burnin", gp_burnin)
                gp_samples = settings.get("gp_samples", gp_samples)
                if opt.gp.chain_ is None:
                    gp_burnin = settings.get("gp_initial_burnin", gp_initial_burnin)
                    gp_samples = settings.get("gp_initial_samples", gp_initial_samples)
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
                logging.warning(
                    "Error encountered during fitting." "Trying to sample chain a bit."
                )
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
