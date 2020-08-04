==================
Chess Tuning Tools
==================


.. image:: https://img.shields.io/pypi/v/chess-tuning-tools.svg
        :target: https://pypi.python.org/pypi/chess-tuning-tools

.. image:: https://img.shields.io/travis/kiudee/chess-tuning-tools.svg
        :target: https://travis-ci.org/kiudee/chess-tuning-tools

.. image:: https://readthedocs.org/projects/chess-tuning-tools/badge/?version=latest
        :target: https://chess-tuning-tools.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A collection of tools for local and distributed tuning of chess engines.


* Free software: Apache Software License 2.0
* Documentation: https://chess-tuning-tools.readthedocs.io.


Features
--------

* Optimization of chess engines using state-of-the-art Bayesian optimization.
* Support for automatic visualization of the optimization landscape.
* Scoring matches using the pentanomial model for paired openings.

Quick Start
-----------

In order to be able to start the tuning, first create a python
environment and install chess-tuning-tools by typing::

   pip install chess-tuning-tools

Furthermore, you need to have `cutechess-cli <https://github.com/cutechess/cutechess>`_
in the path. The tuner will use it to run matches.

To execute the local tuner, simply run::

   tune local -c tuning_config.json

Take a look at the `usage instructions`_ and the `example configurations`_ to
learn how to set up the ``tuning_config.json`` file.


Distributed tuning
------------------

The distributed tuning framework is currently not actively supported.
To be able to run the tuning client, you need the following directory structure::

   folder/
   |---- networks/
   |     |---- networkid
   |---- openings/
   |     |---- ...
   |     |---- openings.pgn
   |     |---- ...
   |---- dbconfig.json
   |---- engine1[.exe]
   |---- engine2[.exe]

Finally, the tuning client can be started as follows::

   cd path/to/folder
   tune run-client dbconfig.json

The client can be terminated gracefully by inputting ctrl-c once or terminated
immediately by sending it twice.

You will also need to run a PostgreSQL database, which the server will use to
post jobs for the clients to fetch and the clients to report their results to.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _example configurations: https://github.com/kiudee/chess-tuning-tools/tree/master/examples
.. _usage instructions: https://chess-tuning-tools.readthedocs.io/en/latest/usage.html
