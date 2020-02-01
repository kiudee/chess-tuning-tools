import psycopg2
import json
import logging
import os
import re
import subprocess
import sys
import numpy as np
from time import sleep
from psycopg2.extras import DictCursor

from .utils import parse_timecontrol, MatchResult, TimeControl

CLIENT_VERSION = 1

__all__ = ["TuningClient"]


class TuningClient(object):
    def __init__(self, dbconfig_path, **kwargs):
        self.logger = logging.getLogger("TuningClient")
        self.lc0_benchmark = None
        self.sf_benchmark = None
        if os.path.isfile(dbconfig_path):
            with open(dbconfig_path, "r") as config_file:
                config = config_file.read().replace("\n", "")
                self.logger.debug(f"Reading DB config:\n{config}")
                self.connect_params = json.loads(config)
        else:
            raise ValueError("No config file found at provided path")

    def run_experiment(self, time_control, cutechess_options):
        try:
            os.remove("out.pgn")
        except OSError:
            pass

        # TODO: For now we assume we always tune against stockfish here
        st = [
            "cutechess-cli",
            "-concurrency",
            f"{cutechess_options['concurrency']}",
            "-engine",
            "conf=lc0",
            f"tc={time_control.engine1}",
            "-engine",
            "conf=sf",
            f"tc={time_control.engine2}",
            "-openings",
            f"file={cutechess_options['opening_path']}",
            "format=pgn",
            "order=random",
            "-draw",
            f"movenumber={cutechess_options['draw_movenumber']}",
            f"movecount={cutechess_options['draw_movecount']}",
            f"score={cutechess_options['draw_score']}",
            "-resign",
            f"movecount={cutechess_options['resign_movecount']}",
            f"score={cutechess_options['resign_score']}",
            "-rounds",
            f"{cutechess_options['rounds']}",
            "-repeat",
            "-games",
            "2",
            # "-tb", "/path/to/tb",  # TODO: Support tablebases
            "-pgnout",
            "out.pgn",
        ]
        out = subprocess.run(st, capture_output=True)
        return self.parse_experiment(out)

    def parse_experiment(self, results):
        string = results.stdout.decode("utf-8")
        err_string = results.stderr.decode("utf-8")
        error_occured = False
        if re.search("connection stalls", string):
            self.logger.error("Connection stalled during match. Aborting client.")
            error_occured = True
        elif re.search("Terminating process", string):
            self.logger.error("Engine was terminated. Aborting client.")
            error_occured = True
        if error_occured:
            self.logger.error(string)
            self.logger.error(err_string)
            sys.exit(1)
        lines = string.split("\n")
        self.logger.debug(f"Cutechess result string:\n{string}")
        last_output = lines[-4]
        result = re.findall(r"[0-9]\s-\s[0-9]\s-\s[0-9]", last_output)[0]
        w, l, d = [float(x) for x in re.findall("[0-9]", result)]
        return MatchResult(wins=w, losses=l, draws=d)

    def run_benchmark(self):
        path = os.path.join(os.path.curdir, "lc0")
        out = subprocess.run([path, "benchmark"], capture_output=True)
        s = out.stdout.decode("utf-8")
        result = float(re.findall(r"([0-9]+\.[0-9]+)\snodes per second", s)[0])
        self.lc0_benchmark = result

        path = os.path.join(os.path.curdir, "sf")
        out = subprocess.run([path, "bench"], capture_output=True)
        # Stockfish outputs results as stderr:
        s = out.stderr.decode("utf-8")
        result = float(re.findall(r"Nodes/second\s+:\s([0-9]+)", s)[0])
        self.sf_benchmark = result

    def adjust_time_control(self, time_control, lc0_nodes, sf_nodes):
        lc0_ratio = lc0_nodes / self.lc0_benchmark
        sf_ratio = sf_nodes / self.sf_benchmark
        tc_lc0 = parse_timecontrol(time_control.engine1)
        tc_sf = parse_timecontrol(time_control.engine2)
        tc_lc0 = [x * lc0_ratio for x in tc_lc0]
        tc_sf = [x * sf_ratio for x in tc_sf]
        # TODO: support non-increment time-control
        return TimeControl(engine1=f"{tc_lc0[0]}+{tc_lc0[1]}", engine2=f"{tc_sf[0]}+{tc_sf[1]}")

    @staticmethod
    def set_working_directories(engine_config):
        path = os.getcwd()
        engine_config[0]["workingDirectory"] = path
        engine_config[1]["workingDirectory"] = path
        if os.name == "nt":  # Windows needs .exe files to work correctly
            engine_config[0]["command"] = "lc0.exe"
            engine_config[1]["command"] = "sf.exe"
        else:
            engine_config[0]["command"] = "./lc0"
            engine_config[1]["command"] = "./sf"

    def run(self):
        while True:
            # 1. Check db for new job

            with psycopg2.connect(**self.connect_params) as conn:
                with conn.cursor(cursor_factory=DictCursor) as curs:
                    job_string = """
                    SELECT * FROM tuning_jobs WHERE active;
                    """
                    curs.execute(job_string)
                    result = curs.fetchall()
                    if len(result) == 0:
                        sleep(30)  # TODO: maybe some sort of decay here
                        continue

                    applicable_jobs = [x for x in result if x["minimum_version"] <= CLIENT_VERSION]
                    if len(applicable_jobs) < len(result):
                        self.logger.warning(
                            "There are jobs which require a higher client version. Please update "
                            "the client as soon as possible!"
                        )
                        if len(applicable_jobs) == 0:
                            sleep(30)
                            continue

                    weights = np.array([x["job_weight"] for x in applicable_jobs])
                    rand_i = np.random.choice(len(applicable_jobs), p=weights / weights.sum())
                    job = applicable_jobs[rand_i]

                    # 2. Set up experiment
                    # a) write engines.json
                    job_id = job["job_id"]
                    config = job["config"]
                    self.logger.debug(f"Received config:\n{config}")
                    engine_config = config["engine"]
                    self.set_working_directories(engine_config)
                    with open("engines.json", "w") as file:
                        json.dump(engine_config, file, sort_keys=True, indent=4)
                    sleep(2)
                    # b) Adjust time control:
                    if self.lc0_benchmark is None:
                        self.logger.info("Running initial nodes/second benchmark to calibrate time controls...")
                        self.run_benchmark()
                        self.logger.info(
                            f"Benchmark complete. Results: lc0: {self.lc0_benchmark} nps, sf: {self.sf_benchmark} nps"
                        )
                    else:
                        self.logger.debug(
                            f"Initial benchmark results: lc0: {self.lc0_benchmark} nps, sf: {self.sf_benchmark} nps"
                        )
                    time_control = self.adjust_time_control(
                        TimeControl(engine1=config["time_control"][0], engine2=config["time_control"][1]),
                        float(job["lc0_nodes"]),
                        float(job["sf_nodes"]),
                    )
                    self.logger.debug(f"Adjusted time control from {config['time_control']} to {time_control}")

                    # 3. Run experiment (and block)
                    self.logger.info(f"Running match with time control\n{time_control}")
                    result = self.run_experiment(time_control=time_control, cutechess_options=config["cutechess"])
                    self.logger.info(f"Match result (WLD): {result.wins} - {result.losses} - {result.draws}")
                    # 5. Send results to database and lock it during access
                    # TODO: Check if job_id is actually present and warn if necessary
                    update_job = """
                    UPDATE tuning_results SET wins = wins + %(wins)s, losses = losses + %(losses)s, draws = draws + %(draws)s
                    WHERE job_id = %(job_id)s;
                    """
                    curs.execute(
                        update_job,
                        {"wins": result.wins, "losses": result.losses, "draws": result.draws, "job_id": job_id},
                    )
                    self.logger.info("Uploaded match result to database.\n")
