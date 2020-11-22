=======
History
=======

0.7.0 (2020-11-22)
------------------
* Fix a bug where the model was not informed about the estimated noise variance
  of the current match.
* Revert default acquisition function back to ``"mes"``.
* Remove noise from the calculation of the confidence interval of the optimum
  value.
* Log cutechess-cli output continuously.
* Add ``"debug_mode"`` parameter which will pass ``-debug`` to cutechess-cli.
* Add support for pondering using ``engineX_ponder``.
* Fix passing boolean UCI options correctly.

0.6.0 (2020-09-20)
------------------
* Add support for input warping, allowing the tuner to automatically transform
  the data into a suitable form (internally).
* Improve default parameters to be slightly more robust for most use cases and
  be more in line with what a user might expect.
* Add confidence interval and standard error of the score of the estimated
  global optimum to the logging output
* Add support for time per move matches (option ``st`` in cutechess-cli).
* Add support for timemargin parameter.
* Fix debug output being spammed by other libraries.
* Fix plots being of varying sizes dependent on their labels and ticks.
  This should make it easier to animate them.

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
