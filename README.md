# PyNEC Python Utilities

This project provides utilities for [Tim's PyNEC](https://github.com/tmolteno/python-necpp/) binding for his [NEC2++](https://github.com/tmolteno/necpp) library.
These utilities includes a class for modelling and simulating the antenna, a class to plot a radiation pattern, and more comming in the future!

# Docs
The documentation is build over at [ReadTheDocs](https://pynec-utilities.readthedocs.io/en/latest/)

Something to note about the documentation is that I am including a pre-built version of PyNEC. This is because the
build on PyPI doesn't include `pyproject.toml`, so the build would fail because NumPy is not installed (and
pip does not follow requirement.txt's order) on ReadTheDocs. This hopefully will be fixed with [PR#21 of python-necpp](https://github.com/tmolteno/python-necpp/pull/21)

# TODO:
- Add axis label for graphs
- Add elevation and azimuth plotting function
- Add more docs
- Add a CLI utility
