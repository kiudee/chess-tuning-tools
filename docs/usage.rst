=====
Usage
=====

Setup
-----

To setup a simple tune, you need the following directory structure:

.. code-block::

   .
   |-- simple_tune.json
   |-- engine1
   |-- engine2
   `-- openings.pgn

``engine1`` and ``engine2`` are the executables for two chess engines.
The convention of this library is that the first engine is the one that will
be tuned.
``openings.pgn`` should contain a suitable set of openings.
Finally, ``simple_tune.json`` is the most important file and contains the
actual tuning configuration. Here we use a very simple configuration:

.. code-block:: JSON

   {
       "engines": [
           {
               "command": "engine1",
               "fixed_parameters": {
                   "Threads": 2
               }
           },
           {
               "command": "engine2",
               "fixed_parameters": {
                   "Threads": 4
               }
           }
       ],
       "parameter_ranges": {
           "CPuct": "Real(0.0, 10.0)"
       },
       "engine1_tc": "1+0.1",
       "engine2_tc": "1+0.1",
       "opening_file": "openings.pgn"
   }

The engines are given as a list. The ``command`` field should contain the
path to the executable of the corresponding engine.
``fixed_parameters`` is optional and can contain fixed UCI parameters for
each engine. These will not be changed by the tuner.
``parameter_ranges`` contains the actual UCI parameters to be optimized
and their ranges. Here we optimize a parameter called ``CPuct``.
We tell the tuner that the parameter can assume real values from 0 to 10.
Another common parameter type is ``Integer``.

An important part of the tuning process is the time control.
The tuner uses
`cutechess-cli <https://github.com/cutechess/cutechess>`_
to run matches. Here we use a time control of 1s + 0.1s increment for
both engines. Depending on the hardware used, it may be necessary to
calibrate the correct ratio of time controls such that both engines are of
equal strength.
The ``opening_file`` field contains the path to an epd or pgn file and will
be used by cutechess-cli to select random openings for the matches.

Run the tuner
-------------

Navigate to the folder and then run:

.. code-block::

   tune local -c simple_tune.json

