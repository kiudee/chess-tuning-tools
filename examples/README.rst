===============
Tuning examples
===============

Quick Start example
-------------------
``simple_tune.json`` contains a simple example tune. You need the following
directory structure:

.. code-block::

   .
   |-- simple_tune.json
   |-- engine1
   |-- engine2
   |-- openings.pgn
   `-- sf.exe

Navigate to the folder and then run:

.. code-block::

   tune local -c simple_tune.json

Complete configuration
----------------------
The file ``complete_config.json`` exhaustively lists all available options and
their default values.
