import psycopg2
import json
import logging
import os
import re
import subprocess
import numpy as np
from collections import namedtuple
from time import sleep
from psycopg2.extras import DictCursor

CLIENT_VERSION = 1


MatchResult = namedtuple("MatchResult", ["wins", "losses", "draws"])
TimeControl = namedtuple("TimeControl", ["engine1", "engine2"])


class TuningClient(object):
    def __init__(self, dbconfig_path, **kwargs):
        self.logger = logging.getLogger("TuningClient")
        if os.path.isfile(dbconfig_path):
            with open(dbconfig_path, "r") as config_file:
                config = config_file.read().replace("\n", "")
                self.logger.info(f"Reading DB config:\n{config}")
                self.connect_params = json.loads(config)
        else:
            raise ValueError("No config file found at provided path")

    def setup_experiment(self):
        pass

    def run_experiment(self, time_control, cutechess_options):
        sleep(5)
        return MatchResult(wins=1, losses=0, draws=0)
        subprocess.run(["rm", "out.pgn"])

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
            # "-repeat",
            "-games",
            "1",  # TODO: Paired openings
            # "-tb", "/path/to/tb",  # TODO: Support tablebases
            "-pgnout",
            "out.pgn",
        ]
        out = subprocess.run(st, capture_output=True)
        return self.parse_experiment(out)

    @staticmethod
    def parse_experiment(results):
        lines = results.stdout.decode("utf-8").split("\n")
        last_output = lines[-4]
        result = re.findall(r"[0-9]\s-\s[0-9]\s-\s[0-9]", last_output)[0]
        w, l, d = [float(x) for x in re.findall("[0-9]", result)]
        return MatchResult(wins=w, losses=l, draws=d)

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
                    print(job)

                    # 2. Set up experiment
                    # a) write engines.json
                    job_id = job["job_id"]
                    config = job["config"]
                    self.logger.debug(f"Received config:\n{config}")
                    engine_config = config["engine"]
                    with open("engines.json", "w") as file:
                        json.dump(engine_config, file, sort_keys=True, indent=4)
                    sleep(2)
                    # b) Adjust time control:
                    # TODO: run benchmark (if necessary) and adjust time control
                    # 3. Run experiment (and block)
                    time_control = TimeControl(engine1=config["time_control"][0], engine2=config["time_control"][1])
                    result = self.run_experiment(time_control=time_control, cutechess_options=config["cutechess"])
                    self.logger.info(f"Match result: {result.wins} - {result.losses} - {result.draws}")
                    # 5. Send results to database and lock it during access
                    update_job = """
                    UPDATE tuning_results SET wins = wins + %(wins)s, losses = losses + %(losses)s, draws = draws + %(draws)s
                    WHERE job_id = %(job_id)s;
                    """
                    curs.execute(
                        update_job,
                        {"wins": result.wins, "losses": result.losses, "draws": result.draws, "job_id": job_id},
                    )
                    self.logger.info("Uploaded match result to database.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tc = TuningClient("dbconfig.json")
    tc.run()
