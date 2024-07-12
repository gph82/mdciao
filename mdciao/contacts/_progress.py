##############################################################################
#    This file is part of mdciao.
#
#    Copyright 2024 Charité Universitätsmedizin Berlin and the Authors
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

import numpy as _np
from time import time as _time
import threading as _threading
from multiprocessing import Manager as _Manager
from IPython.display import display as _display
import ipywidgets as _ipywidgets
from time import sleep as _sleep
from datetime import timedelta as _timedelta
import signal as _signal
from shutil import get_terminal_size as _get_terminal_size

# My attempt at a useful progress reporting for parallel processes.
# The main method is _prepare_progressbar_thread, which generates all objects
# needed to handle asyncronous progress report to the standard output (for the
# terminal or to textwidget in the notebook

def _is_notebook() -> bool:
    # https://stackoverflow.com/a/39662359 and also the answer below (like autonotebook.py)
    try:
        from IPython import get_ipython as _get_ipython #lazy import here is better, can except later
        shell = _get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ModuleNotFoundError):
        return False  # Probably standard Python interpreter

def _progress_dict2infoline(idict, first_update_after=1.5) -> str:
    r"""
    Format the progress-report string from a progress dictionary.

    The progress dictionary is a managed dictionary created by the
    method _prepare_progressbar_thread. It gets passed to the
    individual per-trajectory processes that update its fields as
    they stream through trajectories in parallel (per_traj_ctc and
    per_traj_mindist_lower_bound)

    Meanwhile, a separate thread, which also has access to the
    progress dictionary 1) updates the progress report line using
    this method and takes care of printing w/ or w/o overwriting
    lines in the standard output.

    Both the individual per-trajectory processes (per_traj_ctc and
    per_traj_mindist_lower_bound) as well as the methods that
    called them(trajs2ctcs and  trajs2lower_bounds, respectively)
    used this method to compuse the `infoline`.

    The parameters computed in this method for the `infoline`
    are:
    * n_trajs_done/n_trajs_total
    * Elapsed time
    * remaining time
    * Trajs/s
    * Frames/s

    The `infoline` looks like this:

    `Processing trajectories: 1/4 [ 25%]. Frames processed:    32001. Elapsed time:  0:00:41. Remaining time ~  0:02:04. Trajs/s:    0. Frames/s:  777`

    Please note, the Trajs/s ad Frames/s values are NOT
    instantaneous but overall computation, taking IO
    time into account (measured time is walltime, not
    process time).


    Parameters
    ----------
    idict : dict
        A managed dictionary containing  values needed to
        compile performance parameters to be printed
        in plaintext as a progressbar analog. The dictionary
        needs to have the fields:
        * 'start_time'
        * 'n_trajs_total'
        * 'n_trajs_done'
        * 'n_frames_done'

    first_update_after : float
        In seconds, how much to wait before computing
        performance values (Frames/sec

    Returns
    -------
    infoline : str
        A string with information about the performance
        of the underlying computations. It looks like this:
        `Processing trajectories: 1/4 [ 25%]. Frames processed:    32001. Elapsed time:  0:00:41. Remaining time ~  0:02:04. Trajs/s:    0. Frames/s:  777`
    """
    elapsed = _time() - idict['start_time']
    n = len(str(idict['n_trajs_total']))
    perc = int(_np.round(idict['n_trajs_done'] / idict['n_trajs_total'] * 100))

    if elapsed < first_update_after and idict['n_trajs_done']!=idict["n_trajs_total"]:
        elapsed = 'hh:mm:ss'
        remaining = 'hh:mm:ss'
        trajs_per_s = ''
    else:
        try:
            # Only recompute frames_per_s if more frames have been processed
            # Otherwise if n_frames_done stays the same but the clock is counting,
            # then frames_per_sec necesarily decreases
            if idict["n_frames_done"]>idict["n_frames_done_prev"]:
                idict["frames_per_s"] = int(_np.round(idict['n_frames_done'] / elapsed))
                idict["n_frames_done_prev"]=idict["n_frames_done"]
        except TypeError:
            idict["frames_per_s"] = ''
            idict["n_frames_done"] = ''
        trajs_per_s = int(_np.round(idict['n_trajs_done'] / elapsed))
        try:
            avg_t_traj = elapsed / idict['n_trajs_done']
            remaining = _timedelta(seconds=_np.round(avg_t_traj * (idict['n_trajs_total'] - idict['n_trajs_done'])))
        except ZeroDivisionError:
            remaining = 'hh:mm:ss'
        elapsed = _timedelta(seconds=_np.round(elapsed))
    return f"Processing trajectories: {idict['n_trajs_done'] :{n}}/{idict['n_trajs_total']} [{perc :3}%]. " \
           f"Frames processed: {idict['n_frames_done'] :8}. " \
           f"Elapsed time: {str(elapsed) :>8}. " \
           f"Remaining time ~ {str(remaining) :>8}. " \
           f"Trajs/s: {trajs_per_s  :4}. " \
           f"Frames/s: {idict['frames_per_s'] :4}."

def _progressbardict2thread(progressbar_dict, sleep_between_updates=1, overwrite=True):
    r"""
    Prepares threading objects (an Event and a Thread) as well as the target method of the thread

    Parameters
    ----------
    progressbar_dict : dict
        A managed, thread-safe dict
    sleep_between_updates : float, default is 1
        The frequency at which the progressbar updates
    overwrite : bool, default is True
        Whether to overwrite the output or not. Assumes the
        number of lines needed to be overwritten is
        the same as the lines to be writting.

    Returns
    -------
    thread : _threading.Thread
        A Thread-object with the correct target
        method which knows how to update the
        progress report lines.
    exit_event : _threading.Event
        An event that can be used
        as flag for when to stop updating
    """

    if _is_notebook():
        widg_len = max([len(bar) + 5 for bar in progressbar_dict["pbars"]])
        progress = _ipywidgets.Textarea(value="\n".join(progressbar_dict["pbars"]), rows=len(progressbar_dict["pbars"]),
                                        layout=_ipywidgets.Layout(width=f'{widg_len}ch'),
                                        style={'font-size': f'{16}em'}
                                        )

        # Somewhat from here https://github.com/jupyter-widgets/ipywidgets/issues/2206#issuecomment-483246874
        _display(_ipywidgets.HTML("<style> .no-border textarea { border: none; resize: none; min-width: %uch} </style>"%widg_len))
        progress.add_class("no-border")
        def work(progress):
            while not exit_event.is_set():
                _sleep(sleep_between_updates)
                progressbar_dict["pbars"][0] = _progress_dict2infoline(progressbar_dict)
                progress.layout.width = f'{max([len(bar) + 5 for bar in progressbar_dict["pbars"]])}ch'
                progress.value = "\n".join(progressbar_dict["pbars"])
            if exit_event.is_set():
                # print one last time
                progressbar_dict["pbars"][0] = _progress_dict2infoline(progressbar_dict)
                progress.layout.width = f'{max([len(bar) + 5 for bar in progressbar_dict["pbars"]])}ch'
                progress.value = "\n".join(progressbar_dict["pbars"])

        thread = _threading.Thread(target=work, args=(progress,))
        _display(progress)
    else:
        progress = None
        _print_w_option_to_overwrite(progressbar_dict["pbars"], overwrite=False)
        def work(progress):  # arg is just for compat

            while not exit_event.is_set():
                # Alternatively, one could move the print statements to the individual processes
                # launched by per_traj_ctc (passing a manager.Lock() to per_traj_ctc) s.t.
                # the print is called each time pbars is updated.
                # Instead, here we print at regular sleep(n) intervals
                _sleep(sleep_between_updates)
                progressbar_dict["pbars"][0] = _progress_dict2infoline(progressbar_dict)
                _print_w_option_to_overwrite(progressbar_dict["pbars"], overwrite=overwrite)
            if exit_event.is_set():
                # print one last time
                progressbar_dict["pbars"][0] = _progress_dict2infoline(progressbar_dict)
                _print_w_option_to_overwrite(progressbar_dict["pbars"], overwrite=overwrite)
        thread = _threading.Thread(target=work, args=(progress,))

    exit_event = _threading.Event()
    def handle_kb_interrupt(sig, frame):
        exit_event.set()
        thread.join()
        raise KeyboardInterrupt

    _signal.signal(_signal.SIGINT, handle_kb_interrupt)

    return thread, exit_event

def _print_w_option_to_overwrite(lines, overwrite=False):
    r"""

    Print `lines` with the option to overwrite the previous `len(lines)` lines before printing.

    Tries to take line-wrapping into account

    Todo use a decorator instead of if-else

    Parameters
    ----------
    lines : list
        The lines to be printed, joined
        with the newline "\n" character
    overwrite : bool
        Overwrite the last len(lines) lines,
        by moving the cursor up len(nlines) before
        printing
    """
    if overwrite:
        # Get the terminal size in columns/characters to find out the number of wrapped lines
        terminal_width = _get_terminal_size().columns
        n_wrapped_lines = _np.sum([len(line) > terminal_width for line in lines])
        # ANSI escape characters https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797#escape
        for __ in range(len(lines) + n_wrapped_lines):
            print("\033[1A", end="\r", flush=True)
        print("\n".join(lines), flush=True)
    else:
        print("\n".join(lines), flush=True)

def _prepare_progressbar_thread(progressbar_dict, progressbar):
    r"""

    Prepare the objects needed for a progress report from independent processes
    to report their progress independently of each other.

    These objects are:
    * a managed dictionary, which is thread-safe and can be accessed
      by independent processes and/or threads.
      (https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Manager)
      Apart from key,value pairs, the dictionary itself contains two managed lists:
       * `pbars`, a list strings for the independent progress report lines (one per process called by `delayed`)
       * `indices_of_free_pbars`, a list of booleans to check which `pbars` are free before using them to report
    * a threading.Thread-object, https://docs.python.org/3/library/threading.html#threading.Thread.
      It is instantiated with method called `work` as the `target` parameter (Thread(target=work)). The
      method `work` is created on-the-fly depending on whether we're updating to a terminal (update via print() to stdout)
      or to a notebook (update via a Textwidget).
    * a threading.Event-object, https://docs.python.org/3/library/threading.html#threading.Event.
      It is exit event; the flag that will tell the thread's target method (work) to exit the updating process. I.e.
      work contains a loop that runs with a sleep parameter and a while not exit_event

    The threading objects are created by _progressbardict2thread, which contains the updating logic
    and decisions in its environment dependent `work` method

    Parameters
    ----------
    progressbar_dict : dict
        A normal dictionary containing the fields needed to instantiate a managed
        progressbar dict. These fields are.
        * "n_trajs_total"
        * "n_trajs_done"
        * "n_trajs_done_prev"
        * "n_frames_done"
        * "start_time"
        * "n_jobs"
    progressbar : bool
        Toggle the progressbar. If False, all returned values
        will be None, telling downstream methods not to use
        a progressbar at all. This keeps the logic on whether
        to create a progressbar or not encapsulated here, and
        not in in the other methods.


    Returns
    -------
    progressbar_dict : _Manger().dict() or None
    thread : _threading.Thread or None
    exit_event : _threading.Event or None
    """
    if progressbar:
        # Managed variables (counters) for asynchronous progress bars
        manager = _Manager()
        progressbar_dict = manager.dict(progressbar_dict)
        pbars = manager.list([_progress_dict2infoline(progressbar_dict)] + ["" for ii in range(progressbar_dict["n_jobs"])])
        progressbar_dict.update({"pbars": pbars,
                                 "indices_of_free_pbars": manager.list([False] + [True] * progressbar_dict["n_jobs"])})

        thread, exit_event = _progressbardict2thread(progressbar_dict, sleep_between_updates=0.5)
        thread.start()
    else:
        print("Processing trajectories...", end="\r") #TODO putting this in the main method with decorator
        progressbar_dict, thread, exit_event = None, None, None,


    return progressbar_dict, thread, exit_event