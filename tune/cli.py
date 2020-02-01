"""Console script for chess_tuning_tools."""
import sys
import click
import logging

from .db_workers import TuningServer, TuningClient


@click.group()
def cli():
    pass


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Turn on debug output.')
@click.option('--logfile', default=None, help='Path to where the log is saved to.')
@click.argument('dbconfig')
def run_client(verbose, logfile, dbconfig):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, filename=logfile)
    tc = TuningClient(dbconfig_path=dbconfig)
    tc.run()


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Turn on debug output.')
@click.option('--logfile', default=None, help='Path to where the log is saved to.')
@click.argument('command')
@click.argument('experiment_file')
@click.argument('dbconfig')
def run_server(verbose, logfile, command, experiment_file, dbconfig):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, filename=logfile)
    tc = TuningServer(
        experiment_path=experiment_file,
        dbconfig_path=dbconfig
    )
    if command == 'run':
        tc.run()
    elif command == 'deactivate':
        tc.deactivate()
    elif command == 'reactivate':
        tc.reactivate()
    else:
        raise ValueError(f"Command {command} is not recognized. Terminating...")


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
