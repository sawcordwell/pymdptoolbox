# -*- coding: utf-8 -*-
from distutils.core import setup

setup(name="pymdptoolbox",
      version="4.0a3",
      author="Steven A. W. Cordwell",
      author_email="steven.cordwell@uqconnect.edu.au",
      url="http://code.google.com/p/pymdptoolbox/",
      description="Markov Decision Process (MDP) Toolbox",
      long_description="The MDP toolbox provides classes and functions for "
      "the resolution of descrete-time Markov Decision Processes. The list of "
      "algorithms that have been implemented includes backwards induction, "
      "linear programming, policy iteration, q-learning and value iteration "
      "along with several variations.",
      download_url="http://code.google.com/p/pymdptoolbox/downloads/list",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.3",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      platforms=["Any"],
      license="New BSD",
      
      packages=["mdptoolbox"],
      package_dir={"": "src"},
      requires=["math", "numpy", "scipy", "time"],)
