Markov Decision Process (MDP) Toolbox for Python
================================================

.. image:: https://travis-ci.org/sawcordwell/pymdptoolbox.svg?branch=master
    :target: https://travis-ci.org/sawcordwell/pymdptoolbox
.. image:: https://coveralls.io/repos/sawcordwell/pymdptoolbox/badge.png
  :target: https://coveralls.io/r/sawcordwell/pymdptoolbox

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

Features
--------
  - Eight MDP algorithms implemented
  - Fast array manipulation using `NumPy <http://www.numpy.org>`_
  - Full sparse matrix support using
    `SciPy's sparse package <http://www.scipy.org/SciPyPackages/Sparse>`_
  - Optional linear programming support using
    `cvxopt <http://abel.ee.ucla.edu/cvxopt/>`_

Documentation
-------------
Documentation is available as docstrings in the module code.

.. TODO and as html in the doc folder or from `the MDPtoolbox homepage <>`_.

Installation
------------
NumPy and SciPy must be on your system to use of this toolbox. Please have a
look at their documentation to get them installed. If you are installing
onto Ubuntu or Debian and using Python 2 then this will pull in all the
dependencies:

  ``sudo apt-get install python-numpy python-scipy python-cvxopt``

On the other hand, if you are using Python 3 then cvxopt will have to be
compiled (pip will do it automatically). To get NumPy, SciPy and all the
dependencies to have a fully featured cvxopt then run:

  ``sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev``

I recommend using `pip <https://pip.pypa.io/en/latest/>`_ to install
the toolbox if you have it available. Just type

  ``pip install pymdptoolbox``

at the console and it should take care of downloading and installing everything
for you. If you also want cvxopt to be automatically downloaded and installed
so that you can solve MDPs using linear programming then type:

  ``pip install "pymdptoolbox[LP]"``

If you want it to be installed just for you rather than system wide then do

  ``pip install --user pymdptoolbox``

Otherwise, you can download the package manually from the web

  1. Download the latest stable release from
     https://pypi.python.org/pypi/pymdptoolbox or clone the Git repository

     ``git clone https://github.com/sawcordwell/pymdptoolbox.git``

  2. If you downloaded the `*.zip` or `*.tar.gz` archive, then extract it

     ``tar -xzvf pymdptoolbox-<VERSION>.tar.gz``

     ``unzip pymdptoolbox-<VERSION>``

  3. Change to the PyMDPtoolbox directory

     ``cd pymdptoolbox``

  4. Install via Setuptools, either to the root filesystem or to your home
     directory if you don't have administrative access.

     ``python setup.py install``

     ``python setup.py install --user``
       
     Read the
     `Setuptools documentation <https://pythonhosted.org/setuptools/>`_ for
     more advanced information.

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

