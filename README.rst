
.. image:: https://raw.githubusercontent.com/kiudee/chess-tuning-tools/master/docs/_static/logo.png

|

.. image:: https://img.shields.io/pypi/v/chess-tuning-tools.svg?style=flat-square
        :target: https://pypi.python.org/pypi/chess-tuning-tools

.. image:: https://img.shields.io/travis/com/kiudee/chess-tuning-tools?style=flat-square
        :target: https://travis-ci.com/github/kiudee/chess-tuning-tools

.. image:: https://readthedocs.org/projects/chess-tuning-tools/badge/?version=latest&style=flat-square
        :target: https://chess-tuning-tools.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


A collection of tools for local and distributed tuning of chess engines.


* Free software: Apache Software License 2.0
* Documentation: https://chess-tuning-tools.readthedocs.io.


Features
--------

* Optimization of chess engines using state-of-the-art `Bayesian optimization <https://github.com/kiudee/bayes-skopt>`_.
* Support for automatic visualization of the optimization landscape.
* Scoring matches using a Bayesian-pentanomial model for paired openings.

Quick Start
-----------

In order to be able to start the tuning, first create a python
environment (at least Python 3.7) and install chess-tuning-tools by typing::

   pip install chess-tuning-tools

Furthermore, you need to have `cutechess-cli <https://github.com/cutechess/cutechess>`_
in the path. The tuner will use it to run matches.

To execute the local tuner, simply run::

   tune local -c tuning_config.json

Take a look at the `usage instructions`_ and the `example configurations`_ to
learn how to set up the ``tuning_config.json`` file.


.. _example configurations: https://github.com/kiudee/chess-tuning-tools/tree/master/examples
.. _usage instructions: https://chess-tuning-tools.readthedocs.io/en/latest/usage.html
