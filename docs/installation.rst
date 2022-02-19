.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Chess Tuning Tools, first create a new Python environment with ``python>=3.7``,
then run this command in your terminal:

.. code-block:: console

    $ pip install chess-tuning-tools

This is the preferred method to install Chess Tuning Tools, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

Installation on Windows
^^^^^^^^^^^^^^^^^^^^^^^

To get chess-tuning-tools to work on Windows, the easiest way is to install
the `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ distribution.
Then, create a new environment and install chess-tuning-tools::

   conda create -n myenv python=3.9 scikit-learn=0.23
   activate myenv
   pip install chess-tuning-tools

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Beta release
------------

To give testers early access to new functionality, we will usually release
beta versions to `PyPI`_. To install a beta version, run this command in your
terminal (version is an example):

.. code-block:: console

    $ pip install chess-tuning-tools==0.6.0b2



From sources
------------

The sources for Chess Tuning Tools can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/kiudee/chess-tuning-tools

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/kiudee/chess-tuning-tools/tarball/master

Once you have a copy of the source, you can install it using `poetry`_:

.. code-block:: console

    $ poetry install


.. _Github repo: https://github.com/kiudee/chess-tuning-tools
.. _tarball: https://github.com/kiudee/chess-tuning-tools/tarball/master
.. _poetry: https://python-poetry.org/
.. _PyPI: https://pypi.org/project/chess-tuning-tools/
