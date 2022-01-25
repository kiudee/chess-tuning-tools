"""Console script for chess_tuning_tools."""
import json
import logging
import sys
from datetime import datetime

import click
import dill
import numpy as np
from atomicwrites import AtomicWriter
from skopt.utils import create_result

from tune.db_workers import TuningClient, TuningServer
from tune.io import load_tuning_config, prepare_engines_json, write_engines_json
from tune.local import (
    check_log_for_errors,
    initialize_data,
    initialize_optimizer,
    is_debug_log,
    parse_experiment_result,
    plot_results,
    print_results,
    run_match,
    setup_logger,
    update_model,
)
from tune.priors import create_priors


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
    "--run-only-once",
    default=False,
    is_flag=True,
    help="Terminate the client after one job has been completed or no job can be "
    "found.",
)
@click.option(
    "--skip-benchmark",
    default=False,
    is_flag=True,
    help="Skip calibrating the time control by running a benchmark.",
)
@click.option(
    "--clientconfig", default=None, help="Path to the client configuration file."
)
@click.argument("dbconfig")
def run_client(
    verbose,
    logfile,
    terminate_after,
    run_only_once,
    skip_benchmark,
    clientconfig,
    dbconfig,
):
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
        only_run_once=run_only_once,
        skip_benchmark=skip_benchmark,
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
    "--confidence",
    default=0.90,
    help="Confidence to use for the highest density intervals of the optimum.",
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
    "--gp-signal-prior-scale",
    default=4.0,
    type=click.FloatRange(min=0.0),
    help="Prior scale of the signal (standard deviation) magnitude which is used to"
    "parametrize a half-normal distribution."
    "Needs to be a number strictly greater than 0.0.",
    show_default=True,
)
@click.option(
    "--gp-noise-prior-scale",
    default=0.0006,
    type=click.FloatRange(min=0.0),
    help="Prior scale of the noise (standard deviation) which is used to parametrize a "
    "half-normal distribution."
    "Needs to be a number strictly greater than 0.0.",
    show_default=True,
)
@click.option(
    "--gp-lengthscale-prior-lb",
    default=0.1,
    type=click.FloatRange(min=0.0),
    help="Lower bound for the inverse-gamma lengthscale prior. "
    "It marks the point where the prior reaches 1% of the cumulative density."
    "Needs to be a number strictly greater than 0.0.",
    show_default=True,
)
@click.option(
    "--gp-lengthscale-prior-ub",
    default=0.5,
    type=click.FloatRange(min=0.0),
    help="Upper bound for the inverse-gamma lengthscale prior. "
    "It marks the point where the prior reaches 99% of the cumulative density."
    "Needs to be a number strictly greater than 0.0 and the lower bound.",
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
    default=16,
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
    default=1,
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
    default=1,
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
    "--fast-resume/--no-fast-resume",
    default=True,
    help="If set, resume the tuning process with the model in the file specified by"
    " the --model-path. "
    "Note, that a full reinitialization will be performed, if the parameter"
    "ranges have been changed.",
    show_default=True,
)
@click.option(
    "--model-path",
    default="model.pkl",
    help="The current optimizer will be saved for fast resuming to this file.",
    type=click.Path(exists=False),
    show_default=True,
)
@click.option("--verbose", "-v", count=True, default=0, help="Turn on debug output.")
@click.option(
    "--warp-inputs/--no-warp-inputs",
    default=True,
    show_default=True,
    help="If True, let the tuner warp the input space to find a better fit to the "
    "optimization landscape.",
)
def local(  # noqa: C901
    tuning_config,
    acq_function="mes",
    acq_function_samples=1,
    confidence=0.9,
    data_path=None,
    gp_burnin=5,
    gp_samples=300,
    gp_initial_burnin=100,
    gp_initial_samples=300,
    gp_signal_prior_scale=4.0,
    gp_noise_prior_scale=0.0006,
    gp_lengthscale_prior_lb=0.1,
    gp_lengthscale_prior_ub=0.5,
    logfile="log.txt",
    n_initial_points=16,
    n_points=500,
    plot_every=1,
    plot_path="plots",
    random_seed=0,
    result_every=1,
    resume=True,
    fast_resume=True,
    model_path="model.pkl",
    verbose=0,
    warp_inputs=True,
):
    """Run a local tune.

    Parameters defined in the `tuning_config` file always take precedence.
    """
    json_dict = json.load(tuning_config)
    settings, commands, fixed_params, param_ranges = load_tuning_config(json_dict)
    root_logger = setup_logger(
        verbose=verbose, logfile=settings.get("logfile", logfile)
    )
    root_logger.debug(f"Got the following tuning settings:\n{json_dict}")

    # Initialize/import data structures:
    if data_path is None:
        data_path = "data.npz"
    try:
        X, y, noise, iteration = initialize_data(
            parameter_ranges=list(param_ranges.values()),
            resume=resume,
            data_path=data_path,
        )
    except ValueError:
        root_logger.error(
            "The number of parameters are not matching the number of "
            "dimensions. Rename the existing data file or ensure that the "
            "parameter ranges are correct."
        )
        sys.exit(1)

    # Initialize Optimizer object and if applicable, resume from existing
    # data/optimizer:
    gp_priors = create_priors(
        n_parameters=len(param_ranges),
        signal_scale=settings.get("gp_signal_prior_scale", gp_signal_prior_scale),
        lengthscale_lower_bound=settings.get(
            "gp_lengthscale_prior_lb", gp_lengthscale_prior_lb
        ),
        lengthscale_upper_bound=settings.get(
            "gp_lengthscale_prior_ub", gp_lengthscale_prior_ub
        ),
        noise_scale=settings.get("gp_noise_prior_scale", gp_noise_prior_scale),
    )
    opt = initialize_optimizer(
        X=X,
        y=y,
        noise=noise,
        parameter_ranges=list(param_ranges.values()),
        random_seed=settings.get("random_seed", random_seed),
        warp_inputs=settings.get("warp_inputs", warp_inputs),
        n_points=settings.get("n_points", n_points),
        n_initial_points=settings.get("n_initial_points", n_initial_points),
        acq_function=settings.get("acq_function", acq_function),
        acq_function_samples=settings.get("acq_function_samples", acq_function_samples),
        resume=resume,
        fast_resume=fast_resume,
        model_path=model_path,
        gp_initial_burnin=settings.get("gp_initial_burnin", gp_initial_burnin),
        gp_initial_samples=settings.get("gp_initial_samples", gp_initial_samples),
        gp_priors=gp_priors,
    )

    # Main optimization loop:
    while True:
        root_logger.info("Starting iteration {}".format(iteration))

        # If a model has been fit, print/plot results so far:
        if len(y) > 0 and opt.gp.chain_ is not None:
            result_object = create_result(Xi=X, yi=y, space=opt.space, models=[opt.gp])
            result_every_n = settings.get("result_every", result_every)
            if result_every_n > 0 and iteration % result_every_n == 0:
                print_results(
                    optimizer=opt,
                    result_object=result_object,
                    parameter_names=list(param_ranges.keys()),
                    confidence=settings.get("confidence", confidence),
                )
            plot_every_n = settings.get("plot_every", plot_every)
            if plot_every_n > 0 and iteration % plot_every_n == 0:
                plot_results(
                    optimizer=opt,
                    result_object=result_object,
                    plot_path=settings.get("plot_path", plot_path),
                    parameter_names=list(param_ranges.keys()),
                    confidence=settings.get("confidence", confidence),
                )

        # Ask optimizer for next point:
        point = opt.ask()
        point_dict = dict(zip(param_ranges.keys(), point))
        root_logger.info("Testing {}".format(point_dict))

        # Prepare engines.json file for cutechess-cli:
        engine_json = prepare_engines_json(commands=commands, fixed_params=fixed_params)
        root_logger.debug(f"engines.json is prepared:\n{engine_json}")
        write_engines_json(engine_json, point_dict)

        # Run experiment:
        root_logger.info("Start experiment")
        now = datetime.now()
        out_exp = []
        out_all = []
        for output_line in run_match(**settings):
            line = output_line.rstrip()
            is_debug = is_debug_log(line)
            if is_debug and verbose > 2:
                root_logger.debug(line)
            if not is_debug:
                out_exp.append(line)
            out_all.append(line)
        check_log_for_errors(cutechess_output=out_all)
        out_exp = "\n".join(out_exp)
        later = datetime.now()
        difference = (later - now).total_seconds()
        root_logger.info(f"Experiment finished ({difference}s elapsed).")

        # Parse cutechess-cli output and report results (Elo and standard deviation):
        score, error_variance = parse_experiment_result(out_exp, **settings)
        root_logger.info(
            "Got Elo: {} +- {}".format(-score * 100, np.sqrt(error_variance) * 100)
        )

        # Update model with the new data:
        root_logger.info("Updating model")
        update_model(
            optimizer=opt,
            point=point,
            score=score,
            variance=error_variance,
            acq_function_samples=settings.get(
                "acq_function_samples", acq_function_samples
            ),
            gp_burnin=settings.get("gp_burnin", gp_burnin),
            gp_samples=settings.get("gp_samples", gp_samples),
            gp_initial_burnin=settings.get("gp_initial_burnin", gp_initial_burnin),
            gp_initial_samples=settings.get("gp_initial_samples", gp_initial_samples),
        )

        # Update data structures and persist to disk:
        X.append(point)
        y.append(score)
        noise.append(error_variance)
        iteration = len(X)

        with AtomicWriter(data_path, mode="wb", overwrite=True).open() as f:
            np.savez_compressed(f, np.array(X), np.array(y), np.array(noise))
        with AtomicWriter(model_path, mode="wb", overwrite=True).open() as f:
            dill.dump(opt, f)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
