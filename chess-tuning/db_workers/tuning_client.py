import psycopg2

class TuningClient(object):
    def __init__(self):
        pass

    def run(self):
        while True:
            # 1. Check db for new job
            # 2. Set up experiment
            # 3. Run experiment (and block)
            # 4. Parse result of experiment
            # 5. Send results to database and lock it during access
            pass