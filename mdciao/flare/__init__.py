r"""

Produce *flare-plots*, where the residues are
drawn on a circle and connected with lines of
varying opacity.

It is inspired by the impressive
plots produced by `FlarePlots <https://github.com/GPCRviz/flareplot>`_,
for which you'll find a live-demo
`here <https://gpcrviz.github.io/flareplot/>`_.

Credit should go to `the authors <https://github.com/GPCRviz/flareplot/graphs/contributors>`_
for creating these tools and the beautiful (and useful!) plots.
These tools offer many more functionalities
and have become standard in many applications
and databases.

However, to achieve better compatibility
with the rest of :obj:`mdciao`
and to allow other types of customization,
this **clean-room implementation** in
Python was written from scratch. The
only `special` dependency is for the
`Bezier curves <https://en.wikipedia.org/wiki/B%C3%A9zier_curve>`_,
which is provided by the python module `Bezier <https://github.com/dhermes/bezier>`_.

Note
----
 Flareplots are a type of `Chord-diagrams <https://www.data-to-viz.com/graph/chord.html>`_,
 for which many libraries exist. If you think
 there's a better way of doing what these
 methods are trying to do, please do contact
 me about it!

.. currentmodule:: mdciao.flare

.. autosummary::
   :toctree: generated/
   :nosignatures:

   freqs2flare
   circle_plot_residues
   add_bezier_curves
   add_fragment_labels

"""
from .flare import freqs2flare, circle_plot_residues, add_bezier_curves
from ._utils import add_fragment_labels