Markov Decision Process (MDP) Toolbox for Python
================================================

.. image:: https://travis-ci.org/sawcordwell/pymdptoolbox.svg?branch=master
    :target: https://travis-ci.org/sawcordwell/pymdptoolbox
    :alt: Build Status
.. image:: https://coveralls.io/repos/sawcordwell/pymdptoolbox/badge.png
    :target: https://coveralls.io/r/sawcordwell/pymdptoolbox
    :alt: Code Coverage
.. image:: https://pypip.in/py_versions/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi/pymdptoolbox/
    :alt: Supported Python versions
.. image:: https://pypip.in/implementation/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi/pymdptoolbox/
    :alt: Supported Python implementations
.. image:: https://pypip.in/license/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi/pymdptoolbox/
    :alt: License

The MDP toolbox provides classes and functions for the resolution of
descrete-time Markov Decision Processes. The list of algorithms that have been
implemented includes backwards induction, linear programming, policy iteration,
q-learning and value iteration along with several variations.

The classes and functions were developped based on the
`MATLAB <http://www.mathworks.com/products/matlab/>`_
`MDP toolbox <http://www.inra.fr/mia/T/MDPtoolbox/>`_ by the
`Biometry and Artificial Intelligence Unit <http://mia.toulouse.inra.fr/>`_ of
`INRA Toulouse <http://www.toulouse.inra.fr/>`_ (France). There are editions
available for MATLAB, GNU Octave, Scilab and R.
The suite of MDP toolboxes are described in Chades I, Chapron G, Cros M-J,
Garcia F & Sabbadin R (2014) 'MDPtoolbox: a multi-platform toolbox to solve
stochastic dynamic programming problems', *Ecography*, vol. 37, no. 9, pp.
916–920, doi `10.1111/ecog.00888 <http://dx.doi.org/10.1111/ecog.00888>`_.

Features
--------
  - Eight MDP algorithms implemented
  - Fast array manipulation using `NumPy <http://www.numpy.org>`_
  - Full sparse matrix support using
    `SciPy's sparse package <http://www.scipy.org/SciPyPackages/Sparse>`_
  - Optional linear programming support using
    `cvxopt <http://abel.ee.ucla.edu/cvxopt/>`_

PLEASE NOTE: the linear programming algorithm is currently unavailable except
for testing purposes due to incorrect behaviour.

Installation
------------
NumPy and SciPy must be on your system to use this toolbox. Please have a
look at their documentation to get them installed. If you are installing
onto Ubuntu or Debian and using Python 2 then this will pull in all the
dependencies:

  ``sudo apt-get install python-numpy python-scipy python-cvxopt``

On the other hand, if you are using Python 3 then cvxopt will have to be
compiled (pip will do it automatically). To get NumPy, SciPy and all the
dependencies to have a fully featured cvxopt then run:

  ``sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev``

The two main ways of downloading the package is either from the Python Package
Index or from GitHub. Both of these are explained below.

Python Package Index (PyPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. image:: https://pypip.in/download/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi//pymdptoolbox/
    :alt: Downloads
.. image:: https://pypip.in/version/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi/pymdptoolbox/
    :alt: Latest Version
.. image:: https://pypip.in/status/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi/pymdptoolbox/
    :alt: Development Status
.. image:: https://pypip.in/wheel/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi/pymdptoolbox/
    :alt: Wheel Status
.. image:: https://pypip.in/egg/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi/pymdptoolbox/
    :alt: Egg Status
.. image:: https://pypip.in/format/pymdptoolbox/badge.svg
    :target: https://pypi.python.org/pypi/pymdptoolbox/
    :alt: Download format

The toolbox's PyPI page is https://pypi.python.org/pypi/pymdptoolbox/ and there
are both zip and tar.gz archive options available that can be downloaded.
However, I recommend using `pip <https://pip.pypa.io/en/latest/>`_ to install
the toolbox if you have it available. Just type

  ``pip install pymdptoolbox``

at the console and it should take care of downloading and installing everything
for you. If you also want cvxopt to be automatically downloaded and installed
so that you can help test the linear programming algorithm then type

  ``pip install "pymdptoolbox[LP]"``

If you want it to be installed just for you rather than system wide then do

  ``pip install --user pymdptoolbox``

If you downloaded the package manually from PyPI

  1. Extract the `*.zip` or `*.tar.gz` archive

     ``tar -xzvf pymdptoolbox-<VERSION>.tar.gz``

     ``unzip pymdptoolbox-<VERSION>``

  2. Change to the PyMDPtoolbox directory

     ``cd pymdptoolbox``

  3. Install via Setuptools, either to the root filesystem or to your home
     directory if you don't have administrative access.

     ``python setup.py install``

     ``python setup.py install --user``

     Read the
     `Setuptools documentation <https://pythonhosted.org/setuptools/>`_ for
     more advanced information.

Of course you can also use virtualenv or simply just unpack it to your working
directory.

GitHub
~~~~~~

Clone the Git repository

    ``git clone https://github.com/sawcordwell/pymdptoolbox.git``

and then follow from step two above. To learn how to use Git then I reccomend
reading the freely available `Pro Git book <http://git-scm.com/book>`_ written
by Scott Chacon and Ben Straub and published by Apress.

Quick Use
---------
Start Python in your favourite way. The following example shows you how to
import the module, set up an example Markov decision problem using a discount
value of 0.9, solve it using the value iteration algorithm, and then check the
optimal policy.

.. code:: python

  import mdptoolbox.example
  P, R = mdptoolbox.example.forest()
  vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
  vi.run()
  vi.policy # result is (0, 0, 0)

Documentation
-------------
Documentation is available at http://pymdptoolbox.readthedocs.org/
and also as docstrings in the module code.
If you use `IPython <http://ipython.scipy.org>`_ to work with the toolbox,
then you can view the docstrings by using a question mark ``?``. For example:

.. code:: python

    import mdptoolbox
    mdptoolbox?<ENTER>
    mdptoolbox.mdp?<ENTER>
    mdptoolbox.mdp.ValueIteration?<ENTER>

will display the relevant documentation.

Contribute
----------
Issue Tracker: https://github.com/sawcordwell/pymdptoolbox/issues

Source Code: https://github.com/sawcordwell/pymdptoolbox

Support
-------
Use the issue tracker.

License
-------
The project is licensed under the BSD license. See `<LICENSE.txt>`_ for details.

