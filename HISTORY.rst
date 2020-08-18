=======
History
=======

0.6.0-beta.1 (2020-08-18)
-------------------------
* Add support for time per move matches (option `st` in cutechess-cli).

0.5.0 (2020-08-14)
------------------
* Add support for the new cutechess-cli 1.2.0 output format.
* Add support for confidence intervals of the optimum. By default a table of
  highest density intervals will be reported alongside the current optimum.
* Add support for parameter range reduction. Since this potentially requires
  discarding some of the data points, it will also save a backup.
* Change score calculation to be in logit/Elo space. This fixes problems with
  scores being compressed for very unevenly matched engines.

0.4.0 (2020-08-02)
------------------
* Add new standalone tuning script. With this it is possible to tune parameters
  of an engine without having to set up the distributed tuning framework.
  Usage instructions and example configurations are included.

0.3.0 (2020-03-02)
------------------

* Support for round-flat prior distributions
* Fix parsing of priors and benchmark results

0.2.0 (2020-02-10)
------------------

* Completely new database implemented in SQLAlchemy.
* Pentanomial scoring of matches, accounting for the paired openings and different draw rates of time controls.

0.1.6 (2020-02-02)
------------------

* Allow timed termination of the client by the option ``--terminate-after``

0.1.5 (2020-02-02)
------------------

* Support for non-increment time controls

0.1.4 (2020-02-02)
------------------

* Allow graceful termination of tuning-client using ctrl-c.

0.1.3 (2020-02-01)
------------------

* Implement probabilistic load balancing support in the clients.

0.1.2 (2020-02-01)
------------------

* Simplified tuning client tutorial and logging.

0.1.0 (2020-01-31)
------------------

* First release on PyPI.
