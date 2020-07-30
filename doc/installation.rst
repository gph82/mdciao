.. _installation:

Installation
============

Workshop
--------
For our August workshop  you can download the sources from `here <http://proteinformatics.uni-leipzig.de/mdciao/mdciao-master.zip>`_ and follow the `Installation from source`_.

.. note::
 As almost any Python module, ``mdciao`` comes with some dependencies that will be installed along when installing ``mdciao``. If you don't want ``mdciao`` to alter the existing python installation, we highly recommend to create a separate, virtual python environment to install ``mdciao`` into. For beginners, see the below paragraph


Python versions
---------------
At the moment, ``mdciao`` is CI-tested only for GNU/Linux OSs and Python versions
3.6 and 3.7. See this warning_ for problems during installation from source.

Installation via package manager
--------------------------------
.. warning::
 None of these methods work yet, as ``mdciao`` is not yet published to those package managers. Please use the Installation from the source

We recommend you install ``mdciao`` either via the `pip <https://pypi.org/project/pip/>`_ Python package installer or the `conda <https://conda.io/en/latest/>`_ Python package manager::

 pip install mdciao

or::

 conda install mdciao

Installation from source
------------------------
.. warning::
 * At the moment, the repository is private. For our August workshop  you can download the sources from `here <http://proteinformatics.uni-leipzig.de/mdciao/mdciao-master.zip>`_.
 * Please also see the note on :ref:`hk` if you are planning to run ``mdciao`` on our group cluster.

.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

* |ss| Clone `mdciao's github repository <https://github.com/gph82/mdciao>`_ |se| `Download and unzip the sources <http://proteinformatics.uni-leipzig.de/mdciao/mdciao-master.zip>`_ to your preferred `programs` or `software` folder.

* If you are not familiar with Python environments, please read this `python interpreter and environment`_ note before continuing.

* ``cd`` to the unzipped directory and execute the ``setup.py`` file::

   python setup.py install


This should install ``mdciao`` along with all its dependencies. Optionally using `develop` instead of `install` means that the ``mdciao`` in your Python path points directly to the sources directory, s.t. changes in the source take effect immediately without re-installing

.. warning::
 On some occasions the above command doesn't install `numpy`, `cython` or `mdtraj` properly. Should that happen to you, we recommend issuing::

  pip install cython
  pip install numpy
  pip install mdtraj

 or::

  conda install cython
  conda install numpy
  conda install mdtraj -c conda forge

 **before** installing ``mdciao``.

python interpreter and environment
----------------------------------
`conda <https://docs.conda.io/en/latest/>`_ and `pip <https://pypi.org/project/pip/>`_ are very popular, user friendly package managers. **A very nice feature** of `conda` is that it installs its own ``python`` interpreter, separate from the system's Python. It does so in the user's home directory, s.t. no root privileges are needed.

This means that it's very hard to "break" local Python installations (your own or shared installations, like in clusters). So, don't be afraid to use conda and mess up your Python environment as many times as you like. Wiping and re-installing is easy (delete `~/anaconda3` or `~/miniconda3` from your home directory) and won't not alter your existing Python installation at all!

If you already have conda, and don't want to clutter the ``base`` environment, we recommend you create a new environment::

 conda create -n for_mdciao
 conda activate for_mdciao


If neither `pip` nor `conda` is installed in your system, we recommend you install the bare-bones conda distribution, `miniconda` and build from there:

* Download the latest miniconda from `here <https://docs.conda.io/en/latest/miniconda.html>`_
* Install by issuing::

   sh Miniconda3-latest-Linux-x86_64.sh

and follow the prompt instructions. If you don't want the anaconda Python interpreter to be your default, just answer *no* to the last question.

.. _hk:

Hildiknecht
-----------

.. note::
 If you are on Hildiknecht, `conda` is already installed as module, just issue::

  module load anaconda
  eval "$(conda shell.bash hook)" # if its the first time

 Then you should be able to follow the above instructions no problem!

MacOs and Windows
-----------------

``mdciao`` has been thoroughly tested only in GNU/Linux so far, but you should be able to install and run ``mdciao`` on MacOs/Windows as long as you have a working Python installation and are able to run::

 python setup.py develop

The needed dependencies should install automatically (see above the note about environments) and even if that fails for some reason, you should be able to use *some* package manager to install them manually.

.. toctree::
   :maxdepth: 2
   :hidden:

