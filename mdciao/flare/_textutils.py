##############################################################################
#    This file is part of mdciao.
#    
#    Copyright 2020 Charité Universitätsmedizin Berlin and the Authors
#
#    Authors: Guillermo Pérez-Hernandez
#    Contributors:
#
#    mdciao is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    mdciao is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with mdciao.  If not, see <https://www.gnu.org/licenses/>.
##############################################################################

r"""
These methods attempt to compute the overlaps
between Text objects in a matplotlib plot. These methods
are used by mdciao.flare exclusively, trying to
work around potential re-sizing problems by
trying to avoid re-drawings of the axis (and some shameless
fudging). Naturally, this can be error prone and hard to test.

That's why, for the moment the only tests are that
1) all code is covered by normal execution and nothing goes wrong and
b) visual inspection of the notebooks testing the flareplot schemes

In the future, I'll either understand better how matplotlib
bounding boxes work and/or use libs like shapely to detect
overlaps and stuff.
"""

import numpy as _np

def outermost_text_corner(texts,  center=(0,0), verbose=False,):
    r"""
    Return the distance of the farthest corner of any Text's FancyBoxPatch

    Will fail if:
     * axes.draw() hasn't been called on this axis before
     * text wasn't instantiated with the bbox argument
       Check https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.get_bbox_patch
       for more info on FancyBoxPatch

    Parameters
    ----------
    texts :  list
        :obj:`~matplotlib.text.Text` objects
    center : tuple, default is (0,0)
    verbose : bool, default is False

    Returns
    -------
    d : float

    """
    rmax = 0
    for ii, t in enumerate(texts):
        verts = text2FBPverts(t)
        ir = _np.sqrt(_np.sum((verts-_np.array(center)) ** 2, axis=1)).max()
        if ir>rmax:
            rmax = ir
            idx = ii
    if verbose:
        print("Outermost textbox has index %u and is "%idx, texts[idx])
    return rmax


def plot_fancypatches(texts, lw=5, verbose=False, renderer=None):
    iax = texts[0].axes
    iax.draw(renderer or iax.figure.canvas.get_renderer())
    for t in texts:
        verts = text2FBPverts(t)
        iax.plot(*verts.T, ls="-", lw=lw)
        if verbose:
            print(t)
            print(verts)
            print()

def any_overlap_via_FancyBoxPach(text_objects_1, text_objects_2):
    r"""
    Return the first pair of text objects that overlap

    The texts's FancyBoxPatch is used to check for overlap, since
    it's rotated with the text's orientation. Check
    https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.get_bbox_patch
    for more info on FancyBoxPatch

    The self-overlap is ignored if the lists share members

    Parameters
    ----------
    text_objects_1 : list
        List of :obj:`~matplotlib.text.Text` objects

    text_objects_1 : list
        List of :obj:`~matplotlib.text.Text` objects
    Returns
    -------
    pair : list of two text objects or None
    """

    boxes_2 = {}
    edges_2 = {}
    iax = text_objects_1[0].axes
    renderer = iax.figure.canvas.get_renderer()
    #iax.draw(renderer) # otherwise the Fancyboxes get wrong sizes
    for t1 in text_objects_1:
        t1.draw(renderer)
        edges_1 = text2FBPedges(t1)
        for idx2, t2 in enumerate(text_objects_2):
            if t1 is not t2:
                if idx2 not in boxes_2.keys():
                    t2.draw(renderer)
                    edges_2[idx2] =  text2FBPedges(t2)
                    # draw_line(l2,iax,"b",lw=5)
                    xy = FBintersect_via_edges(edges_1, edges_2[idx2])
                    if xy is not False:
                        # print(l2)
                        #draw_line(l1,iax,"g",lw=5)
                        #t1.axes.plot(xy[0], xy[1], "or", ms=10)
                        #plot_fancypatches([t1, t2], lw=.5)
                        return [t1, t2]

def FBintersect_via_edges(rect1, rect2):
    r"""
    Whether to rectangles intersect

    Rectangles are defined as the edges connecting
    their four corners, i.e. A-B, B-C, C-D, D-A

    Parameters
    ----------
    rect1 : _np.ndarray of shape (4,4)
        Each line of the rect is an
        edge of shape [xA, yA, xB, yB]
        where A and B are the points
        connected by the edge
    rect2 : _np.ndarray of shape (4,4)
        Each line of the rect is an
            edge of shape [xA, yA, xB, yB]
            where A and B are the points
            connected by the edge

    Returns
    -------
    x, y : Tuple or False
        The coordinates of the first intersection
        or False if there is none
    """
    for e1 in rect1:
        for e2 in rect2:
            return  segment_intersection(e1.reshape((2, 2)), e2.reshape((2, 2)))


def text2FBPedges(txt):
    r""" Return, in data units, the edges of the FancyBoxPatch of a Text object

    Parameters
    ----------
    txt : :obj:`~matplotlib.text.Text`

    Returns
    -------
    edges : 2D :obj:`numpy.ndarray` of shape (4,4)
        The edges connecting the vertices of
        the FancyBoxPatch. Each edge consists of
        four floats [xA, yA, xB, yB] where A and
        B are the vertices connected by the edge
    """
    verts_1 = text2FBPverts(txt)
    return _np.vstack([_np.hstack([l1, l2]) for l1, l2 in zip(verts_1[:-1], verts_1[1:])])

def text2FBPverts(txt):
    r"""
    Return, in data units, the vertices of the FancyBoxPatch of a Text object

    Will fail if the Text object wasn't instantiated with a bbox dictionary, because
    then it doesn't have a FancyBoxPatch.

    The advantage of the FancyBoxPatch is that it's rotated with the text
    and it can be used to avoid text-overlap better

    See the matplotlib doc for more info
    https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text

    Parameters
    ----------
    txt : :obj:`~matplotlib.text.Text`

    Returns
    -------
    verts : 2D :obj:`numpy.ndarray` of shape (5,2)
        The vertices of the FancyBoxPatch. Usually,
        the first and last one are the same, so
        that the can be used to create a closed polygonpatch
        in matplotlib
    """
    iax = txt.axes
    txt.update_bbox_position_size(renderer=iax.figure.canvas.get_renderer())
    bbox = txt.get_bbox_patch()
    return iax.transData.inverted().transform(bbox.get_verts())

def draw_line(line, iax,color="k",lw=2):
    r"""

    Parameters
    ----------
    line : x1,y1, x2, y2

    Returns
    -------

    """
    iax.plot((line[0], line[2]), (line[1], line[3]),"-o", color=color, lw=lw)

def line_intersection(line1, line2):
    r"""
    Check if two lines (not segments) intersect

    The line passes through the points (A,B) and (C,D)

    Parameters
    ----------
    line1 : iterable of pairs
        [[xA, yA],[xB, yB]]
    line2 : iterable of pairs
        [[xC, yC],[xD, yD]]

    Returns
    -------
    x, y : tuple
        coordinates of the intersection
        None when there is no intersection
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None
        #raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def segment_intersection(line1,line2):
    r"""
    Check if two segments intersect

    The segments are the line connecting the points (A,B) and (C,D)

    Parameters
    ----------
    line1 : iterable of pairs
        [[xA, yA],[xB, yB]]
    line2 : iterable of pairs
        [[xC, yC],[xD, yD]]

    Returns
    -------
    x, y : tuple or False
        coordinates of the intersection
        False when there is no intersection or
        the intersection lies outside the segments
    """
    xy = line_intersection(line1,line2)
    if xy is not None:
        inside_segments = []
        for line in [line1, line2]:
            vec1, vec2 = xy-line[0], xy-line[1]
            inside_segments.append(_np.dot(vec1,vec2) < 0 )# vectors point in opposite directions
        if all(inside_segments):
            return xy
        else:
            return False
    else:
        return False
