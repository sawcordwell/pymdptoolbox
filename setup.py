# -*- coding: utf-8 -*-

from distutils.core import setup

setup(name="PyMDPtoolbox",
      version="0.9",
      description="Python Markov Decision Problem Toolbox",
      author="Steven Cordwell",
      author_email="steven.cordwell@uqconnect.edu.au",
      url="http://code.google.com/p/pymdptoolbox/",
      license="Modified BSD License",
      py_modules=["mdp"],
      requires=["math", "numpy", "random", "scipy", "time"],)