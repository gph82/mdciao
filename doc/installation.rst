Installation
============
.. note::
 As almost any Python module, `mdciao` comes with some dependencies that will be installed along when installing `mdciao`. If you don't want `mdciao` to alter the existing python installation, we highly recommend to create a separate, virtual python environment to install `mdciao` into. For beginners, see the below paragraph **How to install mdciao on Hildilab**.

Installation via package manager
-----------------------------------


We recommend you install `mdciao` either via the `pip <https://pypi.org/project/pip/>`_ Python package installer or the `conda <https://conda.io/en/latest/>`_ Python package manager::

 pip install mdciao

or::

 conda install mdciao

.. warning::
 None of these methods work yet, as `mdciao` is not yet published to those package managers

Installation from the source
-----------------------------

* You can also clone the sources from `mdciao's github repository <https://github.com/gph82/mdciao>`_ to your preferred `programs` or `software` folder.

.. warning::
 * At the moment, the repository is private.
 * For our workshop on 04.06.2020 you can download the sources from `here <http://proteinformatics.uni-leipzig.de/mdciao/mdciao-develop_tests.zip>`_.
 * Please unzip the sources to your preferred software folder and continue with these instructions:

* Execute the `setup.py` file::

   cd mdciao
   python setup.py develop

This should install `mdciao` along with all its dependencies. The `develop` option means that the `mdciao` in your Python path points directly to the sources directory, s.t. changes in the sources take effect immediately without re-installing.

.. warning::
 At the moment, the `mdtraj` dependency is giving some problems, so we recommend issuing::

  pip install mdtraj

 or::

  conda install mdtraj -c conda forge

 **before** installing `mdciao`.

A note for beginners
---------------------
`conda` and `pip` are very popular, user friendly package managers, but may seem complex to beginners. **A very nice feature** of `conda` is that it installs its own `python` interpreter, separate from the system's Python. It does so in the user's home directory, s.t. no root privileges are needed.

This means that it's very hard to "break" local Python installations (your own or shared installations, like in clusters). So, don't be afraid to use conda and mess up your Python environment as many times as you like. Wiping and re-installing is easy (delete `~/anaconda3` or `~/miniconda3` from your home directory) and won't not alter your existing Python installation at all!

If you already have conda, and don't want to clutter the `base` environment, we recommend you create a new environment::

 conda create -n for_mdciao
 conda activate for_mdciao


If neither `pip` nor `conda` is installed in your system, we recommend you install the bare-bones conda distribution, `miniconda` and build from there:

* Download the latest miniconda from `here <file:///home/guille/Programs/mdciao/doc/_build/html/installation.html>`_
* Install by issuing::

   sh Miniconda3-latest-Linux-x86_64.sh

and follow the prompt instructions. If you don't want the anaconda Python interpreter to be your default, just answer *no* to the last question.

.. note::
 If you are on Hildiknecht, `conda` is already installed as module, just issue::

  module load anaconda
  eval "$(conda shell.bash hook)" # if its the first time

Then you should be able to follow the above instructions no problem!

MacOs and Windows
-----------------

`mdciao` has been thoroughly tested only in GNU/Linux so far, but you should be able to install and run `mdciao` on MacOs/Windows as long as you have a working Python installation and are able to run::

 `python setup.py develop`

The needed dependencies should install automatically (see above the note about environments) and even if that fails for some reason, you should be able to use *some* package manager to install them manually.

