# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 2021
@author: Anton
Version: v0.4-beta


The Subprocessing module contains all functions responsible for the
subprocessing of data.
These are:
    read(), sumforline(), grap2d(), graph3d(), synchronize()
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import conversions as conv


def read(filename: str, delimiter: str = ',', skip_header: int = 1) -> (
        np.ndarray, np.ndarray):
    """
    Reads a .csv file. The first line is skipped. The delimiter ist ','.

    Parameters
    ----------
    filename : str
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


    Since it can happen that two measured values of the time have the same
    value, therefore one of the time points is deleted here. In the vector, an
    average is formed for these two points.
    """
    data = np.genfromtxt(filename, delimiter=delimiter,
                         skip_header=skip_header)
    t = data[:, 2]
    vec = data[:, 3:]
    ts = len(t)
    eq_n = np.array([], dtype=int)
    for n in range(ts-1):
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


def sumforline(filename: str, sub: int = 0) -> int:
    """
    Counts the lines in a .csv file.

    Parameters
    ----------
    filename : str
        File where lines are to be counted
    sub : int, optional
        What should be subtracted from the result.

    Returns
    -------
    res : int
        Number of lines.
    """
    with open(filename) as file:
        res = sum(1 for line in file)

    res -= sub
    return res


def graph2d(t: np.ndarray, y: np.ndarray, typ: str, graph_dict: dict,
            filename: str, string_check: str) -> None:
    """
    Plots a 2d graph.

    Parameters
    ----------
    t : np.ndarray
        The time axis.
    y : np.ndarray
        Size to be poltted.
    graph_dict : dict
        The dictionary which stores all constants for the graph.
    filename : str, optional
        Name of the file used for the creation of the graph
    typ : str, optional
        Units of y.
        Character :  Meaning
          'rot'   :  rotational energy
          'trans' :  translation energy
          else    :  generic labeling
    string_check : str, optional
        See conversions.string

    Returns
    -------
    None.
    """
    _stringf = FormatStrFormatter(graph_dict['formatter'])
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Time in s')
    if 'rot' in typ:
        ax.set_ylabel('$E_{rot}$ in J')
        ax.set_title('rotational energy as a function of time')

    elif 'trans' in typ:
        ax.set_ylabel('$E_{trans}$ in J')
        ax.set_title('translation energy as a function of time')

    elif 'kin' in typ:
        ax.set_ylabel('$E_{kin}$ in J')
        ax.set_title('kenetic energy as a function of time')

    else:
        ax.set_ylabel('y')
        ax.set_title('y as a function of time t' + filename)

    ax.set_ylim(y.min()*0.9, y.max()*1.1)
    ax.set_xlim(t.min()*0.9, t.max()*1.1)
    ax.yaxis.set_major_formatter(_stringf)
    ax.plot(t, y)
    if filename:
        by = conv.string('by ', filename, string_check).replace(",", "")
        fig.text(0.85, 0.9, by, fontsize='x-small')

    ax.grid()
    if graph_dict['save_graph']:
        fname = 'saved_graphs/' + filename + '.png'
        fname = fname.replace('input/', '').replace('.csv', '')
        fig.savefig(fname)


def graph3d(xyz: np.ndarray, graph_dict: dict, string_check: str,
            filename: str = '') -> None:
    """
    Generate a 3d view of the given trajectory.

    Parameters
    ----------
    xyz : np.ndarray
        xyz coordinates of the trajectory.
    graph_dict : dict
        The dictionary which stores all constants for the graph.
    filename : string, optional
        File name for displaying who performed the measurement.
    string_check : string, optional
        See conversions.string

    Returns
    -------
    None.
    """
    _stringf = FormatStrFormatter(graph_dict['formatter'])
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
    if graph_dict['save_graph']:
        fname = conv.string('saved_graphs/', filename, string_check) + '_3d.png'
        fname = fname.replace(',', '')
        fig.savefig(fname)


def synchronize(t_1: np.ndarray, vec_1: np.ndarray, t_2: np.ndarray,
                vec_2: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Synronizes two measurement series so that they have measurement points at
    the same time. The required measurement points are intrapolated.

    Parameters
    ----------
    t_1 : np.ndarray,
        Time of the first measurement series.
    vec_1 : np.ndarray
        Measurement data from t_1.
    t_2 : np.ndarray
        Time of the second measurement series.
    vec_2 : np.ndarray
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
    (vec_1_x, vec_1_y) = vec_1.shape
    (vec_2_x, vec_2_y) = vec_2.shape
    if vec_1_x <= vec_2_x:
        vec_temp = np.zeros((vec_1_x, vec_1_y))
        for n in range(0, vec_1_y):
            vec_temp[:, n] = np.interp(t_1, t_2, vec_2[:, n])
        t = t_1
        vec_2 = vec_temp

    else:
        vec_temp = np.zeros((vec_2_x, vec_2_y))
        for n in range(0, vec_2_y):
            vec_temp[:, n] = np.interp(t_2, t_1, vec_1[:, n])
        t = t_2
        vec_1 = vec_temp

    return (t, vec_1, vec_2)


def input_test(question: str, n: int = 0, n_max: int = 2) -> bool:
    '''
    Ask the user for an input and test whether the input is true or false.

    Parameters
    ----------
    question : str
        Question that the user should answer.
    n : int, optional
        How many times the question has been asked. The default is 0.
    n_max : int, optional
        How many times the question may be answered maximum ambiguous.

    Raises
    ------
    ValueError
        If the maximum number of ambiguous entries has been reached.

    Returns
    -------
    bool
        User response.

    '''
    test = input(f'{question} ')
    if test in ['False', 'false', '0', 'No', 'no', 'N', 'n']:
        res = False

    elif test in ['True', 'true', '1', 'Yes', 'yes', 'Y', 'y']:
        res = True

    else:
        if n < n_max:
            n += 1
            print('Valid inputs are: y for yes and n for no.')
            print(f'{test} is not a valid input.')
            input_test(question, n=n, n_max=n_max)

        else:
            raise ValueError(f'{test} is not a valid input.')

    print('')
    return res


def filename_sorting_key(filename: str) -> int:
    '''
    Returns a value for the given file name by which it can be sorted.

    Parameters
    ----------
    filename : str
        The name of the file to be sorted.

    Returns
    -------
    key : int
        Sort key for the given name.

    '''
    if 'AccGyr' in filename:
        key = 0
    elif 'Accelerometer' in filename:
        key = 1
    elif 'Gyroscope' in filename:
        key = 2
    elif 'LinearAcceleration' in filename:
        key = 4
    elif 'Quaterion' in filename:
        key = 8
    elif 'Magnetometer' in filename:
        key = 16
    elif 'Gravity' in filename:
        key = 32
    elif 'AmbientLight' in filename:
        key = 64
    elif 'Pressure' in filename:
        key = 128
    elif 'Temperature' in filename:
        key = 256
    else:
        key = 32768

    return key
