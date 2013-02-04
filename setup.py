# -*- coding: utf-8 -*-

from distutils.core import setup

setup(name="PyMDPtoolbox",
      version="4.0alpha1",
      description="Python Markov Decision Problem Toolbox",
      author="Steven Cordwell",
      author_email="steven.cordwell@uqconnect.edu.au",
      url="http://code.google.com/p/pymdptoolbox/",
      license="New BSD License",
      py_modules=["mdp"],
      requires=["math", "numpy", "random", "scipy", "time"],)
