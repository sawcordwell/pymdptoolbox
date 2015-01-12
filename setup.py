# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name="pymdptoolbox",
      version="4.0-b4-dev",
      author="Steven A. W. Cordwell",
      author_email="steven.cordwell@uqconnect.edu.au",
      url="https://github.com/sawcordwell/pymdptoolbox",
      description="Markov Decision Process (MDP) Toolbox",
      long_description="The MDP toolbox provides classes and functions for "
      "the resolution of descrete-time Markov Decision Processes. The list of "
      "algorithms that have been implemented includes backwards induction, "
      "linear programming, policy iteration, q-learning and value iteration "
      "along with several variations.",
      download_url="https://pypi.python.org/pypi/pymdptoolbox",
      classifiers=[
          "Development Status :: 4 - Beta",
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
          "Programming Language :: Python :: 3.2",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      platforms=["Any"],
      license="New BSD",

      packages=find_packages("src", exclude=["tests"]),
      package_dir={"": "src"},
      install_requires=["numpy", "scipy"],
      extras_require={"LP": "cvxopt"})
