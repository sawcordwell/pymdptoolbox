Markov Decision Process (MDP) Toolbox 4.0 for Python
====================================================

The MDP toolbox provides classes and functions for the resolution of
descrete-time Markov Decision Processes. The list of algorithms that have been
implemented includes backwards induction, linear programming, policy iteration,
q-learning and value iteration along with several variations.

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

    3. Change to the MDP toolbox directory 
       ``cd pymdptoolbox``

    4. Install via Distutils, either to the filesystem or to a home directory
       ``python setup.py install``
       ``python setup.py install --home=<dir>``

Quick Use
---------
Start Python in your favourite way. Then follow the example below to import the
module, set up an example Markov decision problem using a discount value of 0.9,
and solve it using the value iteration algorithm.

    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.policy
    (0, 0, 0)

