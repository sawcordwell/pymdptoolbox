Markov Decision Process (MDP) Toolbox 4.0 for Python
====================================================

The MDP toolbox provides classes and functions for the resolution of
descrete-time Markov Decision Processes. The list of algorithms that have been
implemented includes backwards induction, linear programming, policy iteration,
q-learning and value iteration along with several variations.

The classes and functions were developped based on the
`MATLAB <http://www.mathworks.com/products/matlab/>`_ 
`MDP toolbox http://www.inra.fr/mia/T/MDPtoolbox/>`_ by the
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
Documentation is available as docstrings in the module code and as html in the
doc folder or from `the MDPtoolbox homepage <http://www.>`_.

Installation
------------
    1. Download the latest stable release from 
       `http://code.google.com/p/pymdptoolbox/downloads/list`_  or clone the
       Git repository
       ``git clone https://code.google.com/p/pymdptoolbox/``

    2. If you downloaded the `*.zip` or `*.tar.gz` archive, then extract it
       ``tar -xzvf pymdptoolbox-<VERSION>.tar.gz``
       ``unzip pymdptoolbox-<VERSION>``

    3. Change to the PyMDPtoolbox directory 
       ``cd pymdptoolbox``

    4. Install via Distutils, either to the filesystem or to a home directory
       ``python setup.py install``
       ``python setup.py install --home=<dir>``

Quick Use
---------
Start Python in your favourite way. Then follow the example below to import the
module, set up an example Markov decision problem using a discount value of 0.9,
solve it using the value iteration algorithm, and then check the optimal policy.

    >>> import mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> vi.policy
    (0, 0, 0)

Contribute
----------
Issue Tracker: https://code.google.com/p/pymdptoolbox/issues/list
Source Code: https://code.google.com/p/pymdptoolbox/source/checkout

Support
-------

License
-------
The project is licensed under the BSD license. See LICENSE.txt for details.

