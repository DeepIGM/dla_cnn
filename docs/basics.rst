.. highlight:: rest

******
Basics
******

Here are some basics of the DLA CNN code.

Dataset
========

The code is intended to be run/trained on a set of
spectra.  We will refer to these as the dataset.

The dataset is now held in a *Data* object.

Catalog
+++++++

Each dataset has a catalog that summarizes
the set of spectra.  This must include the
source redshift.   This catalog is held
within the *Data* object.

ID
++

The pointer to each spectrum is held in
an *Id* class.  It is a very simple object.

Sightline
+++++++++

The data and *Id* for a given spectrum is held
in the *Sightline* object.  This class also includes
a method to process the data with the CNN, although
a large batch of Sightlines should be processed in a more
efficient manner.
