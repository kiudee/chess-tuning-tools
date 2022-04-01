=======
History
=======

0.9.3 (2022-04-01)
------------------

Local tuner
~~~~~~~~~~~
- Add support for twosided resign adjudication. If set to true, both engines
  have to report an absolute score higher than the ``resign_score`` threshold
  (#95).
- Add support for ``engineX_restart`` (default "auto") which allows one to set
  the restart mode used by cutechess (#95).
- Add depth-based time control using ``engineX_depth`` (#95).
- Log the estimated draw rate of the current match (#197).


0.9.2 (2022-03-13)
------------------

Local tuner
~~~~~~~~~~~

- Add CLI flag ``--evaluate-points -p CSVFILE`` which allows the user to supply
  a CSV file with points the tuner should try before resuming normal operation.
  An optional integer column can be specified, which will be used to indicate
  the number of rounds to run each point for (#91).
- Log version of the tuner in the log output (#191).

0.9.1 (2022-03-06)
------------------

Local tuner
~~~~~~~~~~~

- Fix a bug where the Elo and optima plots were showing incorrect iterations,
  if the user reduced parameter ranges (#183).
- Only use scientific notation in the optima plots, if the parameter is
  optimized on a log scale (#182).

0.9.0 (2022-02-11)
------------------

Local tuner
~~~~~~~~~~~

- Add a plot which shows the optima predicted by the tuner across the
  iterations (#172). This can be useful to gauge convergence.
- Add a plot which shows the estimated Elo (+ confidence interval) of the
  predicted optima (#176).
- Tuner saves optima and their Elo performance (including standard deviation)
  to disk now (#171).

0.8.3 (2022-01-30)
------------------

- Fix compatibility with Python 3.7 (#150).

0.8.2 (2022-01-26)
------------------

Local tuner
~~~~~~~~~~~

* Add support for plotting one-dimensional optimization landscapes (#34).

0.8.1 (2021-09-11)
------------------

Local tuner
~~~~~~~~~~~

* Emit errors and warnings for common cutechess-cli problems:

  - Unknown UCI parameters
  - An engine loses on time
  - The connection to an engine stalls (usually a crash)
  - Any other error captured by cutechess-cli
* Add support for specifying nested parameters.

0.8.0 (2021-08-15)
------------------

Local tuner
~~~~~~~~~~~

* Replace default lengthscale priors by inverse-gamma distributions.
* Add the following command line flags, which allow the user to override the
  prior parameters:

  - ``--gp-signal-prior-scale`` for the scale of the signal prior.
  - ``--gp-noise-prior-scale`` for the scale of the noise prior.
  - ``--gp-lengthscale-prior-lb`` for the lower bound of the lengthscale prior.
  - ``--gp-lengthscale-prior-ub`` for the upper bound of the lengthscale prior.

0.7.3 (2021-06-27)
------------------

Local tuner
~~~~~~~~~~~

* Add ``--fast-resume`` switch to the tuner, which allows instant resume
  functionality from disk (new default).
* Fix the match parser producing incorrect results, when concurrency > 1 is
  used for playing matches.

Distributed tuning framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The distributed tuning framework is no longer deprecated.
* Add ``--run-only-once`` flag to distributed tuning client. If True, it will
  terminate after completing one job or immediately if no job is found.
* Add ``--skip-benchmark`` flag to distributed tuning client. If True, it will
  skip the calibration of the time control, which involves running a benchmark
  for both engines.
* Tuning server of the distributed tuning framework will now also save the
  optimizer object.
* Tuning server now also uses the updated pentanomial model including
  noise estimation.
* ``warp_inputs`` can now be passed via database to the tuning server.
* Fix the server for distributed tuning not sorting the data by job id causing
  the model to be fit with randomly permuted scores.
* Fix the server for distributed tuning trying to compute the current optimum
  before a model has been fit.

0.7.2 (2021-03-22)
------------------
* Print user facing scores using the more common Elo scale, instead of negative
  downscaled values used internally.
* Internal constants set to improved values.
* Always send ``uci`` first before sending ``setoption`` commands to the engine.

0.7.1 (2020-12-08)
------------------
* Fix incorrectly outputting the variance instead of the standard deviation for
  the estimated error around the score estimate.

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
