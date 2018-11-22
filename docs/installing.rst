.. highlight:: rest

**********
Installing
**********

This document describes how to install code
required for the `qso_lya_detection_pipeline`
repository.  We also describe
:ref:`download-public`.

Installing Dependencies
=======================
We have and will continue to keep the number of dependencies low.
There are a few standard packages and a few not-so-standard
packages that need to be installed for this code.

In general, we recommend that you use Anaconda for the majority of
these installations.

Detailed installation instructions are presented below:

Python Dependencies
-------------------

specdb depends on the following list of Python packages.

We recommend that you use `Anaconda <https://www.continuum.io/downloads/>`_
to install and/or update these packages.

* `python <http://www.python.org/>`_ version 3.6 or later
* `numpy <http://www.numpy.org/>`_ version 1.14 or later
* `astropy <http://www.astropy.org/>`_ version 3.0 or later
* `scipy <http://www.scipy.org/>`_ version 1.0 or later
* `matplotlib <http://matplotlib.org/>`_  version 2.0 or later
* `PyQT5 <https://wiki.python.org/moin/PyQt/>`_ version 5 (needed for linetools)
* `h5py <https://www.h5py.org/>`_ version 2.6 (for data I/O)
* `h5py <https://www.h5py.org/>`_ version 2.6 (for data I/O)
* `tensorflow <https://www.tensorflow.org/>`_ version X.X
* `peakutils <https://peakutils.readthedocs.io/en/latest/>`_ version 1.3
* `fasteners <https://pypi.org/project/fasteners/>`_ version 0.14

If you are using Anaconda, you can check the presence of these packages with::

	conda list "^python|numpy|astropy|scipy|matplotlib|pyqt|h5py"

If the packages have been installed, this command should print
out all the packages and their version numbers.

If any of these packages are missing you can install most
of them with a command like::

	conda install h5py

These require pip::

    pip install peakutils

If any of the packages are out of date, they can be updated
with a command like::

	conda update scipy
	pip update peakutils


Installing qso_lya_detection_pipeline
=====================================

Presently, you must download the code from github::

	#go to the directory where you would like to install dlacnn.
	git clone https://github.com/davidparks21/qso_lya_detection_pipeline

From there, you can build and install with

	cd qso_lya_detection_pipeline
	python setup.py install  # or use develop


This should install the package and scripts.
Make sure that your PATH includes the standard
location for Python scripts (e.g. ~/anaconda/bin)



