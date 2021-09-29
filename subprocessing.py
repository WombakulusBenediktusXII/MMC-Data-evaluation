# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:31:14 2021
@author: Anton
Version: v0.3-alpha


The Subprocessing module contains all functions responsible for the
subprocessing of data.
These are:
    read(), sumforline(), grap2d(), graph3d(), synchronize()
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import conversions as conv


def read(filename: str, delimiter: str = ',',
         skip_header: int = 1) -> np.ndarray:
    """
    Reads a .csv file. The first line is skipped. The delimiter ist ','.

    Parameters
    ----------
    filename : str, mandatory
        The name of the file to be read in.
    delimiter : str, optional
        By what character the individual data points are separated
        from each other. The default is ','.
    skip_header : int, optional
        How many lines to skip at the beginning. The default is 1.

    Returns
    -------
    t : np.ndarray
        Time of measurement for the individual measuring points
    vec : np.ndarray
        Measured values

    Comment:
        Since it can happen that two measured values of the time have the same
        value, therefore one of the time points is deleted here. In the vector,
        an average is formed for these two points.
    """
    if filename == '':
        raise TypeError('No filename has been specified.')
    data = np.genfromtxt(filename, delimiter=delimiter,
                         skip_header=skip_header)
    t = data[:, 2]
    vec = data[:, 3:]
    ts = len(t)
    eq_n = np.array([], dtype=int)
    for n in np.arange(0, ts-1):
        if t[n] == t[n+1]:
            eq_n = np.append(eq_n, n)
    if vec.ndim != 1:
        for n in eq_n:
            vec[n, :] = (vec[n, :] + vec[n+1, :]) / 2
    else:
        for n in eq_n:
            vec[n] = (vec[n] + vec[n+1]) / 2

    t = np.delete(t, eq_n+1)
    vec = np.delete(vec, eq_n+1, axis=0)
    return (t, vec)


def sumforline(filename: str) -> int:
    """
    Counts the lines in a .csv file.

    Parameters
    ----------
    filename : str, mandatory
        File where lines are to be counted

    Returns
    -------
    sum : int
        Number of lines.
    """
    if filename == '':
        raise TypeError('No filename has been specified.')
    with open(filename) as file:
        return sum(1 for line in file)


def graph2d(t: np.ndarray, y: np.ndarray, typ: str, filename: str,
            string_check: str, formatter: str = '%1.2e') -> None:
    """
    Plots a 2d graph.

    Parameters
    ----------
    t : np.ndarray, mandatory
        The time axis.
    y : np.ndarray, mandatory
        Size to be poltted.
    typ : str, optional
        Units of y.
        Character :  Meaning
          'rot'   :  rotational energy
          'trans' :  translation energy
          else    :  generic labeling

    formatter : str, optional
        How many digits are displayed on the y-axis. The default is '%1.2e'.
    filename : str, optional
        Name of the file used for the creation of the graph
    string_check : str, optional
        See conversions.string

    Returns
    -------
    None.

    """
    _stringf = FormatStrFormatter(formatter)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Time in s')
    if 'rot' in typ:
        ax.set_ylabel('$E_{rot}$ in J')
        ax.set_title('rotational energy as a function of time')
    elif 'trans' in typ:
        ax.set_ylabel('$E_{trans}$ in J')
        ax.set_title('translation energy as a function of time')
    else:
        ax.set_ylabel('y')
        ax.set_title('y as a function of time t' + filename)
    ax.set_ylim(y.min()*0.9, y.max()*1.1)
    ax.set_xlim(t.min()*0.9, t.max()*1.1)
    ax.yaxis.set_major_formatter(_stringf)
    ax.plot(t, y)
    if filename != '':
        by = conv.string('by ', filename, string_check).replace(",", "")
        fig.text(0.85, 0.9, by, fontsize='x-small')
    ax.grid()
    return None


def graph3d(xyz: np.ndarray, string_check: str,  formatter: str = '%1.2e',
            filename: str = '') -> None:
    """
    Generate a 3d view of the given trajectory.

    Parameters
    ----------
    xyz : np.ndarray, mandatory
        xyz coordinates of the trajectory.
    formatter : string, optional
         How many digits are displayed on the xyz-axis. The default is '%1.2e'.
    filename : string, optional
        File name for displaying who performed the measurement.
    string_check : string, optional
        See conversions.string

    Returns
    -------
    None.

    """
    _stringf = FormatStrFormatter(formatter)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x in m')
    ax.set_ylabel('y in m')
    ax.set_zlabel('z in m')
    ax.set_title('sensor trajectory')
    ax.set_xlim(xyz[:, 0].min()*0.9, xyz[:, 0].max()*1.1)
    ax.set_ylim(xyz[:, 1].min()*0.9, xyz[:, 1].max()*1.1)
    ax.set_zlim(xyz[:, 2].min()*0.9, xyz[:, 2].max()*1.1)
    ax.xaxis.set_major_formatter(_stringf)
    ax.yaxis.set_major_formatter(_stringf)
    ax.zaxis.set_major_formatter(_stringf)
    ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    if filename:
        by = conv.string('by ', filename, string_check).replace(",", "")
        fig.text(0.85, 0.9, by, fontsize='x-small')

    ax.grid()
    return None


def synchronize(t_1: np.ndarray, vec_1: np.ndarray, t_2: np.ndarray,
                vec_2: np.ndarray) -> np.ndarray:
    """
    Synronizes two measurement series so that they have measurement points at
    the same time. The required measurement points are intrapolated.

    Parameters
    ----------
    t_1 : np.ndarray,, mandatory
        Time of the first measurement series.
    vec_1 : np.ndarray, mandatory
        Measurement data from t_1.
    t_2 : np.ndarray, mandatory
        Time of the second measurement series.
    vec_2 : np.ndarray, mandatory
        Measurement data from t_2.

    Returns
    -------
    t : np.ndarray
        Synchronous time.
    vec_1 : np.ndarray
        Synchronous measurement series one.
    vec_2 : np.ndarray
        Synchronous measurement series two.
    """
    (vec_1_len, _) = vec_1.shape
    (vec_2_len, _) = vec_2.shape
    if vec_1_len <= vec_2_len:
        len_ = vec_1_len
    else:
        len_ = vec_2_len

    t = t_2[:len_]
    vec_1 = vec_1[:len_, :]
    vec_2 = vec_2[:len_, :]
    return (t, vec_1, vec_2)
