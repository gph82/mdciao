.. _installation:

Installation
============

Installation via package manager
--------------------------------

.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>


We recommend you install ``mdciao`` via the `pip <https://pypi.org/project/pip/>`_ Python package installer::

 pip install mdciao

Installation via the `conda <https://conda.io/en/latest/>`_ Python package manager is not ready yet.

.. note::
 As almost any Python module, ``mdciao`` comes with some dependencies that will be installed along when installing ``mdciao``. If you don't want ``mdciao`` to alter the existing python installation, we highly recommend to create a separate, virtual python environment to install ``mdciao`` into. More info on how to do this in the note about the `Python interpreter and environment`_.

.. warning::
 If you are interested in latest ``mdciao`` features, please use the `Installation from source`_.


Installation from source
------------------------

.. note::
 If you are not familiar with Python environments, please read this `Python interpreter and environment`_ note before continuing.

* Clone or download `mdciao's github repository <https://github.com/gph82/mdciao>`_ to your preferred ``programs`` or ``software`` folder. If you are using a terminal and have   `git <https://git-scm.com/downloads>`_ installed, simply: ::

   git clone https://github.com/gph82/mdciao.git


  Cloning with ``git`` will allow you to easily get fixes and new features if you *pull* regularly. If you don't have `git <https://git-scm.com/downloads>`_, you can use `wget <https://www.gnu.org/software/wget/>`_ (or MacOs equivalent) to simply download a *snapshot* of the repository at its current status (you'll have to re-dowload again every time to get fixes and new features)::

   wget https://github.com/gph82/mdciao/archive/master.zip

  and if you don't have ``wget`` simply browse to `mdciao's github repository <https://github.com/gph82/mdciao>`_ and download from there via your browser.

* If you are not familiar with Python environments, please read this `Python interpreter and environment`_ note before continuing.

* ``cd`` to the (unzipped) ``mdciao`` directory and `install from the local source files <https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-a-local-src-tree>`_::

   python3 -m pip install .

  This should install ``mdciao`` along with all its dependencies. Also, you can use: ::

   python3 -m pip install -e .

  Adding the option `-e` or `--editable`, means that the ``mdciao`` in your Python path points directly to the sources directory, s.t. changes in the source take effect immediately without re-installing

.. _warning:
.. warning::
 On some occasions the above commands don't install `numpy`, `cython` or `mdtraj` properly. Should that happen to you, we recommend issuing::

  pip install cython
  pip install numpy
  pip install mdtraj

 or::

  conda install cython
  conda install numpy
  conda install mdtraj -c conda forge

 **before** installing ``mdciao``.

Operating systems and Python versions
-------------------------------------
``mdciao`` is developed in GNU/Linux, and CI-tested via `github actions <https://github.com/gph82/mdciao/actions?query=workflow%3A%22Python+package%22>`_ for GNU/Linux and MacOs. Tested python versions are:

* GNU/Linux: 3.7, 3.8, 3.9, 3.10
* MacOs: 3.7, 3.8, 3.9

So everything should work *out of the box* in these conditions. Please see this warning_ for problems during installation from source.

.. note::
 MacOs CI tests sometimes fail with (semi)random segmentation faults, in particular when using mdtraj and compute_dssp, but are in principle sane. (`example of recently passed test <https://github.com/gph82/mdciao/actions/runs/2648833037>`_, `example of recently failed tests <https://github.com/gph82/mdciao/runs/7272124587?check_suite_focus=true>`_). More the details on the segmentation error (139 and 11) when `calling mdtraj.dsssp can be found here <https://github.com/gph82/mdciao/runs/2415051993?check_suite_focus=true>`_.


Python interpreter and environment
----------------------------------
`conda <https://docs.conda.io/en/latest/>`_ and `pip <https://pypi.org/project/pip/>`_ are very popular, user friendly package managers. **A very nice feature** of `conda` is that it installs its own ``python`` interpreter, separate from the system's Python. It does so in the user's home directory, s.t. no root privileges are needed at any moment.

This means that it's very hard to "break" local Python installations (your own or shared installations, like in clusters). So, don't be afraid to use conda and mess up your Python environment as many times as you like. Wiping and re-installing individual environments is easy (`conda remove -n my_test_env --all`), same with entire conda installations (delete `~/anaconda3` or `~/miniconda3` from your home directory). None of this will alter your OS-wide Python installation at all!

If you already have ``conda``, and don't want to clutter the ``base`` environment, we recommend you create a new environment::

 conda create -n for_mdciao
 conda activate for_mdciao

If you prefer ``pip``, please see their documentation on `creating a virtual environment <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment>`_.

If neither ``pip`` nor ``conda`` is installed in your system, we recommend you install the bare-bones conda distribution, ``miniconda`` and build from there:

* Download the latest miniconda from `here <https://docs.conda.io/en/latest/miniconda.html>`_
* Install by issuing::

   sh Miniconda3-latest-Linux-x86_64.sh

and follow the prompt instructions. If you don't want the anaconda Python interpreter to be your default, just answer *no* to the last question.