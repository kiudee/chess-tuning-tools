"""Console script for chess_tuning_tools."""
import sys
import click
import logging

from .db_workers import TuningServer, TuningClient


@click.group()
def cli():
    pass


@cli.command()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Turn on debug output.")
@click.option("--logfile", default=None, help="Path to where the log is saved to.")
@click.option("--terminate-after", default=0, help="Terminate the client after x minutes.")
@click.option("--clientconfig", default=None, help="Path to the client configuration file.")
@click.argument("dbconfig")
def run_client(verbose, logfile, terminate_after, clientconfig, dbconfig):
    """ Run the client to generate games for distributed tuning.

    In order to connect to the database you need to provide a valid DBCONFIG
    json file. It contains the necessary parameters to connect to the database
    where it can fetch jobs and store results.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, filename=logfile, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    tc = TuningClient(dbconfig_path=dbconfig, terminate_after=terminate_after, clientconfig=clientconfig)
    tc.run()


@cli.command()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Turn on debug output.")
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
        level=log_level, filename=logfile, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
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


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
