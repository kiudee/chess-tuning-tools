import json
import logging
import os
import re
import signal
import subprocess
import sys
import numpy as np
from time import sleep, time

try:
    from sqlalchemy.orm import sessionmaker
except ImportError:
    sessionmaker = None

from tune.db_workers.dbmodels import (
    Base,
    SqlJob,
    SqlUCIParam,
    SqlResult,
    SqlTimeControl,
    SqlTune,
)
from tune.io import InitStrings
from tune.db_workers.utils import (
    MatchResult,
    TimeControl,
    create_sqlalchemy_engine,
    get_session_maker,
)
from tune.utils import parse_timecontrol

CLIENT_VERSION = 2

__all__ = ["TuningClient"]


class TuningClient(object):
    def __init__(self, dbconfig_path, terminate_after=0, clientconfig=None, **kwargs):
        self.end_time = None
        if terminate_after != 0:
            start_time = time()
            self.end_time = start_time + terminate_after * 60
        self.logger = logging.getLogger("TuningClient")
        self.lc0_benchmark = None
        self.sf_benchmark = None
        signal.signal(signal.SIGINT, self.interrupt_handler)
        self.interrupt_pressed = False
        if os.path.isfile(dbconfig_path):
            with open(dbconfig_path, "r") as config_file:
                config = config_file.read().replace("\n", "")
                self.logger.debug(f"Reading DB config:\n{config}")
                self.connect_params = json.loads(config)
        else:
            raise ValueError(f"No config file found at provided path:\n{dbconfig_path}")

        self.engine = create_sqlalchemy_engine(self.connect_params)
        Base.metadata.create_all(self.engine)
        sm = sessionmaker(bind=self.engine)
        self.sessionmaker = get_session_maker(sm)

        self.client_config = None
        if clientconfig is not None:
            if os.path.isfile(clientconfig):
                with open(clientconfig, "r") as ccfile:
                    config = ccfile.read().replace("\n", "")
                    self.client_config = json.loads(config)
            else:
                raise ValueError(
                    f"Client configuration file not found:\n{clientconfig}"
                )

    def interrupt_handler(self, sig, frame):
        if self.interrupt_pressed:
            self.logger.info("Shutting down immediately.")
            sys.exit(0)
        self.interrupt_pressed = True
        self.logger.info(
            "Signal received. Shutting down after next match.\nPress a second time to terminate immediately."
        )

    def run_experiment(self, time_control, cutechess_options):
        try:
            os.remove("out.pgn")
        except OSError:
            pass

        st = [
            "cutechess-cli",
            "-concurrency",
            f"{cutechess_options['concurrency']}",
            "-engine",
            f"conf=engine1",
            f"tc={time_control.to_strings()[0]}",
            "-engine",
            "conf=engine2",
            f"tc={time_control.to_strings()[1]}",
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
            "-pgnout",
            "out.pgn",
        ]
        if "syzygy_path" in cutechess_options:
            st.insert(-2, "-tb")
            st.insert(-2, cutechess_options["syzygy_path"])
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

    def run_benchmark(self, config):
        def uci_to_cl(k, v):
            mapping = {
                "Threads": "--threads",
                "NNCacheSize": "--nncache",
                "Backend": "--backend",
                "BackendOptions": "--backend-opts",
                "MinibatchSize": "--minibatch-size",
                "MaxPrefetch": "--max-prefetch",
            }
            if k in mapping:
                return f"{mapping[k]}={v}"
            return None

        def cl_arguments(init_strings):
            cl_args = []
            for k, v in init_strings.items():
                arg = uci_to_cl(k, v)
                if arg is not None:
                    cl_args.append(arg)
            return cl_args

        self.logger.debug(
            f"Before benchmark engine 1:\n{config['engine'][0]['initStrings']}"
        )
        args = cl_arguments(InitStrings(config["engine"][0]["initStrings"]))
        self.logger.debug(f"Arguments for engine 1: {args}")
        path = os.path.join(os.path.curdir, "lc0")
        out = subprocess.run([path, "benchmark"] + args, capture_output=True)
        s = out.stdout.decode("utf-8")
        try:
            result = float(re.findall(r"([0-9\.]+)\snodes per second", s)[0])
        except IndexError:
            self.logger.error(f"Error while parsing engine1 benchmark:\n{s}")
            sys.exit(1)
        self.lc0_benchmark = result

        self.logger.debug(
            f"Before benchmark engine 2:\n{config['engine'][1]['initStrings']}"
        )
        num_threads = InitStrings(config["engine"][1]["initStrings"])["Threads"]
        path = os.path.join(os.path.curdir, "sf")
        out = subprocess.run(
            [path, "bench", "16", str(int(num_threads))], capture_output=True
        )
        # Stockfish outputs results as stderr:
        s = out.stderr.decode("utf-8")
        try:
            result = float(re.findall(r"Nodes/second\s+:\s([0-9]+)", s)[0])
        except IndexError:
            self.logger.error(f"Error while parsing engine2 benchmark:\n{s}")
            sys.exit(1)
        self.sf_benchmark = result

    def adjust_time_control(self, time_control, lc0_nodes, sf_nodes):
        lc0_ratio = lc0_nodes / self.lc0_benchmark
        sf_ratio = sf_nodes / self.sf_benchmark
        new_tc = TimeControl(
            engine1_time=float(time_control.engine1_time) * lc0_ratio,
            engine1_increment=float(time_control.engine1_increment) * lc0_ratio,
            engine2_time=float(time_control.engine2_time) * sf_ratio,
            engine2_increment=float(time_control.engine2_increment) * sf_ratio,
        )
        return new_tc

    @staticmethod
    def set_working_directories(job, engine_config):
        path = os.getcwd()
        engine_config[0]["workingDirectory"] = path
        engine_config[1]["workingDirectory"] = path
        exe1 = job.engine1_exe
        exe2 = job.engine2_exe
        if os.name == "nt":  # Windows needs .exe files to work correctly
            engine_config[0]["command"] = f"{exe1}.exe"
            engine_config[1]["command"] = f"{exe2}.exe"
        else:
            engine_config[0]["command"] = f"./{exe1}"
            engine_config[1]["command"] = f"./{exe2}"

    def pick_job(self, rows, mix=0.25):
        weights = []
        results = []
        min_samples = []
        for job, result in rows:
            weights.append(float(job.weight))
            results.append(result)
            min_samples.append(job.minimum_samplesize)
        """Pick a job based on weight and current load."""
        weights = np.array(weights)
        self.logger.debug(f"Job weights: {weights}")
        sample_size = np.array(
            [
                x.ww_count
                + x.wd_count
                + x.wl_count
                + x.dd_count
                + x.dl_count
                + x.ll_count
                for x in results
            ]
        )
        self.logger.debug(f"Sample sizes: {sample_size}")
        min_samples = np.array(min_samples)
        missing = np.maximum(min_samples - sample_size, 0.0)
        self.logger.debug(f"Missing samples: {missing}")
        if np.all(missing == 0.0):
            p = np.ones_like(weights) * weights
            p /= p.sum()
        else:
            uniform = np.ones_like(weights) / len(weights)
            p = missing * weights
            p /= p.sum()
            p = mix * uniform + (1 - mix) * p
        self.logger.debug(f"Resulting p={p}")
        rand_i = np.random.choice(len(results), p=p)
        result = results[rand_i]
        self.logger.debug(
            f"Picked result {rand_i} (job_id={result.job_id}, tc_id={result.tc_id})"
        )
        return rows[rand_i]

    def incorporate_config(self, config):
        admissible_uci = [
            "Threads",
            "Backend",
            "BackendOptions",
            "NNCacheSize",
            "MinibatchSize",
            "MaxPrefetch",
        ]
        engines = config["engine"]
        for i, e in enumerate(engines):
            engine_str = f"engine{i+1}"
            if engine_str in self.client_config:
                init_strings = InitStrings(e["initStrings"])
                for k, v in self.client_config[engine_str].items():
                    if k in admissible_uci:
                        init_strings[k] = v
        if "SyzygyPath" in self.client_config:
            path = self.client_config["SyzygyPath"]
            for e in engines:
                init_strings = InitStrings(e["initStrings"])
                init_strings["SyzygyPath"] = path
            config["cutechess"]["syzygy_path"] = path

    def run(self):
        while True:
            if self.interrupt_pressed:
                self.logger.info("Shutting down after receiving shutdown signal.")
                sys.exit(0)
            if self.end_time is not None and self.end_time < time():
                self.logger.info("Shutdown timer triggered. Closing")
                sys.exit(0)
            # 1. Check db for new job
            with self.sessionmaker() as session:
                rows = (
                    session.query(SqlJob, SqlResult)
                    .filter(SqlJob.id == SqlResult.job_id, SqlJob.active == True)
                    .all()
                )
                if len(rows) == 0:
                    sleep(30)  # TODO: maybe some sort of decay here
                    continue

                applicable_jobs = [
                    row for row in rows if row[0].minimum_version <= CLIENT_VERSION
                ]
                if len(applicable_jobs) < len(rows):
                    self.logger.warning(
                        "There are jobs which require a higher client version. Please update "
                        "the client as soon as possible!"
                    )
                    if len(applicable_jobs) == 0:
                        sleep(60)
                        continue
                job, sql_result = self.pick_job(applicable_jobs)

                # 2. Set up experiment
                # a) write engines.json
                config = json.loads(job.config)
                if self.client_config is not None:
                    self.incorporate_config(config)
                # TODO: remove: job_id = job["job_id"]
                self.logger.debug(f"Received config:\n{config}")
                engine_config = config["engine"]
                self.set_working_directories(job, engine_config)
                with open("engines.json", "w") as file:
                    json.dump(engine_config, file, sort_keys=True, indent=4)
                sleep(2)
                # b) Adjust time control:
                if self.lc0_benchmark is None:
                    self.logger.info(
                        "Running initial nodes/second benchmark to calibrate time controls."
                        "Ensure that your pc is idle to get a good reading."
                    )
                    self.run_benchmark(config)
                    self.logger.info(
                        f"Benchmark complete. Results: lc0: {self.lc0_benchmark} nps, sf: {self.sf_benchmark} nps"
                    )
                else:
                    self.logger.debug(
                        f"Initial benchmark results: lc0: {self.lc0_benchmark} nps, sf: {self.sf_benchmark} nps"
                    )
                orig_tc = sql_result.time_control.to_tuple()
                time_control = self.adjust_time_control(
                    orig_tc, float(job.engine1_nps), float(job.engine2_nps),
                )
                self.logger.debug(f"Adjusted time control from {orig_tc} to {time_control}")

                # 3. Run experiment (and block)
                self.logger.info(f"Running match with time control\n{time_control}")
                result = self.run_experiment(
                    time_control=time_control, cutechess_options=config["cutechess"]
                )
                self.logger.info(
                    f"Match result (WLD): {result.wins} - {result.losses} - {result.draws}"
                )
                # 5. Send results to database and lock it during access
                q = session.query(SqlResult).filter_by(
                        job_id=job.id, tc_id=sql_result.time_control.id
                    )
                if result.wins == 2:  # WW
                    q.update({"ww_count": SqlResult.ww_count + 1})
                elif result.wins == 1:
                    if result.draws == 1:  # WD
                        q.update({"wd_count": SqlResult.wd_count + 1})
                    else:  # WL
                        q.update({"wl_count": SqlResult.wl_count + 1})
                elif result.draws == 2:  # DD
                    q.update({"dd_count": SqlResult.dd_count + 1})
                elif result.draws == 1:  # DL
                    q.update({"dl_count": SqlResult.dl_count + 1})
                else:  # LL
                    q.update({"ll_count": SqlResult.ll_count + 1})
                self.logger.info("Uploaded match result to database.\n")
