=====================================
Running chess-tuning-tools on vast.ai
=====================================

`vast.ai <https://vast.ai/>`_ is a cloud GPU rental service where hosts can
offer their GPU machines and other users can rent them. Due to the cheap prices,
it is a popular platform in the LeelaChessZero community to run self-play
experiments, matches and tunes.
This tutorial will explain how to set up a basic tuning run for
LeelaChessZero.

This folder contains:

* ``config.json``: The configuration file for the tuning script.
* ``onstart.sh``: The setup script which will install all the dependencies,
  set up the folder structure and run the tuning script. In addition,
  it will resume the tuning script, if your vast.ai instance is restarted.
* ``README.rst``: This document.

-----
Setup
-----

config.json
^^^^^^^^^^^

We will start with the ``config.json`` file. The entry ``"engines"`` contains
a list of engines. The first one is the one whose parameters will be tuned:

.. code-block:: json

   {
       "command": "lc0",
       "fixed_parameters": {
           "Threads": 2
       }
   },

``command`` here is the command to call the engine.
``fixed_parameters`` contains a dictionary of fixed UCI parameters. With that
we can fix any parameters of the engine, which are not supposed to be modified
during tuning.

The most important part of the tuning configuration is the ``parameter_ranges``
block:

.. code-block:: json

   "parameter_ranges": {
       "CPuct": "Real(0.0, 5.0)",
       "CPuctAtRoot": "Real(0.0, 5.0)"
   },

With that we tell the optimizer which parameters are to be optimized, what type
of parameter it is and what the valid range is. Here we have two real-valued
parameters in the interval ``[0, 5]``.
For some parameters it can be useful to optimize them in log-space.
This can be done by specifying ``Real(0.01, 10.0, prior='log-uniform')``.
Another important parameter type is ``Integer``, which can be set as follows:
``Integer(1, 1024)``.

Other important settings you might want to change:

* ``engineX_tc``: Time control to use for the engine. Follows the same
  format used in ``cutechess-cli``.
* ``rounds``: Number of rounds (2 games each) to play to evaluate one
  configuration. The fewer rounds, the more noisy each evaluation will be,
  but each evaluation will be much faster.
  By default a compromise of 5 rounds (10 games) is used. If you run very short
  games, then it is better to increase the number of rounds to 30-50.
* ``opening_file``: The file containing the openings. Can be pgn or epd.
* ``tb_path`` and ``adjudicate_tb``: If you installed tablebases, that is where
  you can set the path.

Once you are finished, it is convenient to upload the file to cloud storage like
`Gist`_. That way the ``onstart.sh`` script can download and start it
automatically.


onstart.sh
^^^^^^^^^^
There are usually not many things you need to change here. The file itself is
documented with comments. Here is a list of the things you might want to change:

* The specific LeelaChessZero version (maybe you are optimizing the parameters
  for a pull request).
* Download your ``config.json`` file.
* Download the 3-5 piece Syzygy tablebases.
* Download other openings

------------------
Running the script
------------------

1. Allocate a new vast.ai instance using the
   ``nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04`` image.
2. Transfer your ``onstart.sh`` to the home folder root of our vast.ai instance::

      scp -P <instance-port> onstart.sh root@<instance-ip>:~/

3. ssh to your instance and run::

      $ chmod +x onstart.sh
      $ ./onstart.sh

   This will install all dependencies and set up the folder structure.
4. (optional) Run benchmarks or calibration matches.
5. Run ``./onstart.sh`` again if you set everything up. It will now execute the
   tuning script.

The script will save a log file to ``~/tuning/log.txt`` and output plots
to ``~/tuning/plots/``.
If you are using an interruptable instance, the script will be run
automatically, if your instance is restarted and will resume the tune.


.. _Gist: https://gist.github.com/
