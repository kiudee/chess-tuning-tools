"""Console script for chess_tuning_tools."""
from datetime import datetime
import json
import logging
import os
import pathlib
import sys
import time

from atomicwrites import AtomicWriter
from bask.optimizer import Optimizer
import click
import matplotlib.pyplot as plt
import numpy as np
from skopt.utils import create_result


from tune.db_workers import TuningServer, TuningClient
from tune.local import run_match, parse_experiment_result, reduce_ranges
from tune.io import prepare_engines_json, load_tuning_config, write_engines_json
from tune.plots import plot_objective
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
    "-c",
    "--tuning-config",
    help="json file containing the tuning configuration.",
    required=True,
    type=click.File("r"),
)
@click.option(
    "-a",
    "--acq-function",
    default="mes",
    help="Acquisition function to use for selecting points to try. "
    "Can be {mes, pvrs, ei, ts, vr}.",
    show_default=True,
)
@click.option(
    "--acq-function-samples",
    default=1,
    help="How many GP samples to average the acquisition function over. "
    "More samples will slow down the computation time, but might give more "
    "stable tuning results. Less samples on the other hand cause more exploration "
    "which could help avoid the tuning to get stuck.",
    show_default=True,
)
@click.option(
    "-d",
    "--data-path",
    default="data.npz",
    help="Save the evaluated points to this file.",
    type=click.Path(exists=False),
    show_default=True,
)
@click.option(
    "--gp-burnin",
    default=5,
    type=int,
    help="Number of samples to discard before sampling model parameters. "
    "This is used during tuning and few samples suffice.",
    show_default=True,
)
@click.option(
    "--gp-samples",
    default=300,
    type=int,
    help="Number of model parameters to sample for the model. "
    "This is used during tuning and it should be a multiple of 100.",
    show_default=True,
)
@click.option(
    "--gp-initial-burnin",
    default=100,
    type=int,
    help="Number of samples to discard before starting to sample the initial model "
    "parameters. This is only used when resuming or for the first model.",
    show_default=True,
)
@click.option(
    "--gp-initial-samples",
    default=300,
    type=int,
    help="Number of model parameters to sample for the initial model. "
    "This is only used when resuming or for the first model. "
    "Should be a multiple of 100.",
    show_default=True,
)
@click.option(
    "-l",
    "--logfile",
    default="log.txt",
    help="Path to log file.",
    type=click.Path(exists=False),
    show_default=True,
)
@click.option(
    "--n-initial-points",
    default=30,
    help="Size of initial dense set of points to try.",
    show_default=True,
)
@click.option(
    "--n-points",
    default=500,
    help="The number of random points to consider as possible next point. "
    "Less points reduce the computation time of the tuner, but reduce "
    "the coverage of the space.",
    show_default=True,
)
@click.option(
    "--plot-every",
    default=0,
    help="Plot the current optimization landscape every n-th iteration. "
    "Set to 0 to turn it off.",
    show_default=True,
)
@click.option(
    "--plot-path",
    default="plots",
    help="Path to the directory to which the tuner will output plots.",
    show_default=True,
)
@click.option(
    "--random-seed",
    default=0,
    help="Number to seed all internally used random generators.",
    show_default=True,
)
@click.option(
    "--result-every",
    default=5,
    help="Output the actual current optimum every n-th iteration."
    "The further you are in the tuning process, the longer this will take to "
    "compute. Consider increasing this number, if you do not need the output "
    "that often. Set to 0 to turn it off.",
    show_default=True,
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Let the optimizer resume, if it finds points it can use.",
    show_default=True,
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
    plot_every=5,
    plot_path="plots",
    random_seed=0,
    result_every=5,
    resume=True,
    verbose=False,
):
    """Run a local tune.

    Parameters defined in the `tuning_config` file always take precedence.
    """
    json_dict = json.load(tuning_config)
    settings, commands, fixed_params, param_ranges = load_tuning_config(json_dict)
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    file_logger = logging.FileHandler(settings.get("logfile", logfile))
    file_logger.setFormatter(log_format)
    root_logger.addHandler(file_logger)
    console_logger = logging.StreamHandler(sys.stdout)
    console_logger.setFormatter(log_format)
    root_logger.addHandler(console_logger)
    logging.debug(f"Got the following tuning settings:\n{json_dict}")

    # 1. Create seed sequence
    ss = np.random.SeedSequence(settings.get("random_seed", random_seed))
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
            if len(X[0]) != opt.space.n_dims:
                logging.error(
                    "The number of parameters are not matching the number of dimensions. "
                    "Rename the existing data file or ensure that the parameter ranges are correct."
                )
                sys.exit(1)
            reduction_needed, X_reduced, y_reduced, noise_reduced = reduce_ranges(
                X, y, noise, opt.space
            )
            if reduction_needed:
                backup_path = path.parent / (
                    path.stem + f"_backup_{int(time.time())}" + path.suffix
                )
                logging.warning(
                    f"The parameter ranges are smaller than the existing data. "
                    f"Some points will have to be discarded. "
                    f"The original {iteration} data points will be saved to {backup_path}"
                )
                np.savez_compressed(
                    backup_path, np.array(X), np.array(y), np.array(noise)
                )
                X = X_reduced
                y = y_reduced
                noise = noise_reduced

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
        result_every_n = settings.get("result_every", result_every)
        if (
            result_every_n > 0
            and iteration % result_every_n == 0
            and opt.gp.chain_ is not None
        ):
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
        plot_every_n = settings.get("plot_every", plot_every)
        if (
            plot_every_n > 0
            and iteration % plot_every_n == 0
            and opt.gp.chain_ is not None
        ):
            logging.getLogger("matplotlib.font_manager").disabled = True
            if opt.space.n_dims == 1:
                logging.warning(
                    "Plotting for only 1 parameter is not supported " "yet."
                )
            else:
                logging.debug("Starting to compute the next plot")
                result_object = create_result(
                    Xi=X, yi=y, space=opt.space, models=[opt.gp]
                )
                plt.style.use("dark_background")
                fig, ax = plt.subplots(
                    nrows=opt.space.n_dims,
                    ncols=opt.space.n_dims,
                    figsize=(3 * opt.space.n_dims, 3 * opt.space.n_dims),
                )
                fig.patch.set_facecolor("#36393f")
                for i in range(opt.space.n_dims):
                    for j in range(opt.space.n_dims):
                        ax[i, j].set_facecolor("#36393f")
                timestr = time.strftime("%Y%m%d-%H%M%S")
                plot_objective(
                    result_object, dimensions=list(param_ranges.keys()), fig=fig, ax=ax
                )
                plotpath = pathlib.Path(settings.get("plot_path", plot_path))
                plotpath.mkdir(parents=True, exist_ok=True)
                full_plotpath = plotpath / f"{timestr}-{iteration}.png"
                plt.savefig(
                    full_plotpath,
                    pad_inches=0.1,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="#36393f",
                )
                logging.info(f"Saving a plot to {full_plotpath}")
                plt.close(fig)
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

        if os.name == "nt":
            np.savez_compressed(data_path, np.array(X), np.array(y), np.array(noise))
        else:
            with AtomicWriter(data_path, mode="wb", overwrite=True).open() as f:
                np.savez_compressed(f, np.array(X), np.array(y), np.array(noise))


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
