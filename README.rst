
.. image:: https://raw.githubusercontent.com/kiudee/chess-tuning-tools/master/docs/_static/CTT-Plots.png

.. image:: https://raw.githubusercontent.com/kiudee/chess-tuning-tools/master/docs/_static/logo.png

.. image:: https://readthedocs.org/projects/chess-tuning-tools/badge/?version=latest&style=flat-square
        :target: https://chess-tuning-tools.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/234719111.svg?style=flat-square
   :target: https://zenodo.org/badge/latestdoi/234719111


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

Installation on Windows
^^^^^^^^^^^^^^^^^^^^^^^

To get chess-tuning-tools to work on Windows, the easiest way is to install
the `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ distribution.
Then, create a new environment and install chess-tuning-tools::

   conda create -n myenv python=3.9 scikit-learn=0.23
   activate myenv
   pip install chess-tuning-tools

.. _example configurations: https://github.com/kiudee/chess-tuning-tools/tree/master/examples
.. _usage instructions: https://chess-tuning-tools.readthedocs.io/en/latest/usage.html
