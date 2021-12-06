# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 2021
@author: Anton
Version: v0.4-beta

The Processing module contains all functions responsible for the direct
processing of raw data.
These are:
    main(), accelerometer(), gyroscope(), accgyr(), failed(), str_gen()
"""

import time
import numpy as np

import subprocessing as sub
import conversions as conv


def main(filename: str, acc_dict: dict, gyr_dict: dict,
         graph_dict: dict) -> tuple:
    '''
    Selects the correct analysis method.

    Parameters
    ----------
    filename : str
        Name of the file being processed.
    acc_dict : dict
        The dictionary which stores all constants for the accelerometer.
    gyr_dict : dict
        The dictionary which stores all constants for the gyroscope.
    graph_dict : dict, madatory
        The dictionary which stores all constants for the graph.

    Returns
    -------
    data : tuple
        The calculated energies for the processed file.

    '''
    if 'Accelerometer' in filename:
        (E_trans, t) = accelerometer(filename, acc_dict, graph_dict)
        data = (filename, t, E_trans, None, None)

    elif 'Gyroscope' in filename:
        (E_rot, t) = gyroscope(filename, gyr_dict, graph_dict)
        data = (filename, t, None, E_rot, None)

    elif 'AccGyr' in filename:
        (E_trans, E_rot,
         E_kin, t) = accgyr(filename, acc_dict, gyr_dict, graph_dict)
        data = (filename, t, E_trans, E_rot, E_kin)

    else:
        failed(filename)
        data = (filename, None, None, None, None)

    return data


def accelerometer(filename: str, acc_dict: dict, graph_dict: dict) -> (
        np.ndarray, np.ndarray):
    """
    Read the file of the accelerometer and calculate the speed from it.
    From this the translations energy is calculated and output as a
    diagram over time.

    Parameters
    ----------
    filename : str
        The name of the file to be evaluated.
    acc_dict : dict
        The dictionary which stores all constants for the accelerometer.
    graph_dict : dict, madatory
        The dictionary which stores all constants for the graph.

    Returns
    -------
    E_trans : np.ndarray
        Translational energy.
    t : np.ndarray
        Time of measurement for the individual translation energies.
    """
    time_local_start = time.perf_counter()
    print(f'{filename}: ', end='')
    (t, a) = sub.read(filename)
    (v, t_step) = conv.velocity(a=a, t=t, acc_dict=acc_dict)
    E_trans = 0.5 * acc_dict['m'] * (v[:, 0]**2+v[:, 1]**2+v[:, 2]**2)
    if graph_dict['do_graph']:
        sub.graph2d(t=t, y=E_trans, typ='trans', filename=filename,
                    string_check='A', graph_dict=graph_dict)
        if acc_dict['trajectory']:
            xyz = conv.xyz(t_step, v)
            sub.graph3d(xyz=xyz, filename=filename, string_check='A',
                        graph_dict=graph_dict)

    time_local_end = time.perf_counter()
    time_local = round((time_local_end - time_local_start), 3)
    print(f'took {time_local}s to process.')
    return (E_trans, t)


def gyroscope(filename: str, gyr_dict: dict, graph_dict: dict) -> (np.ndarray,
                                                                   np.ndarray):
    """
    Reads the file of the gyroscope. The rotation energy is then calculated
    from this. BUT it is calculated for a solid full sphere, so for everything
    but a sphere very poorly suited. The rotational energy is output as a graph
    over time.

    Parameters
    ----------
    filename : str
        The name of the file to be evaluated.
    gyr_dict : dict
        The dictionary which stores all constants for the gyroscope.
    graph_dict : dict, madatory
        The dictionary which stores all constants for the graph.

    Returns
    -------
    E_rot : np.ndarray
        Rotational energy.
    t : np.ndarray
        Time of measurement for the individual translation energies.
    """
    time_local_start = time.perf_counter()
    print(f'{filename}: ', end='')
    (t, rot_raw) = sub.read(filename)
    (rot_vel, _, _) = conv.rotation(rot_raw, t, rot_mode='v', gyr_dict=gyr_dict)
    omega = np.sqrt(rot_vel[:, 0]**2 + rot_vel[:, 1]**2 + rot_vel[:, 2]**2)
    E_rot = 0.4 * gyr_dict['m'] * (gyr_dict['r']**2) * omega**2
    if graph_dict['do_graph']:
        sub.graph2d(t=t, y=E_rot, typ='rot', filename=filename,
                    string_check='G', graph_dict=graph_dict)

    time_local_end = time.perf_counter()
    time_local = round((time_local_end - time_local_start), 3)
    print(f'took {time_local}s to process.')
    return (E_rot, t)


def accgyr(filename: str, acc_dict: dict, gyr_dict: dict, graph_dict: dict) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Reads the data from the gyroscope and the accelerometer. Calculated the
    rotational energy and the translational energy and subtracted the angular
    acceleration from the measured acceleration.

    Parameters
    ----------
    filename : str
        The name of the file to be evaluated.
    acc_dict : dict
        The dictionary which stores all constants for the accelerometer.
    gyr_dict : dict
        The dictionary which stores all constants for the gyroscope.
    graph_dict : dict, madatory
        The dictionary which stores all constants for the graph.

    Returns
    -------
    E_trans : np.ndarray
        Translational energy.
    E_rot : np.ndarray
        Rotational energy.
    E_kin : np.ndarray
        Kenetic energy
    t : np.ndarray
        Time of measurement for the individual energies.
    """
    time_local_start = time.perf_counter()
    sensorname = filename.replace("_AccGyr.csv", "").replace("input/", "")
    print(f'From {sensorname} the gyroscope and accelerometer: ', end='')
    filename_gyr = f'input/{sensorname}_Gyroscope.csv'
    (t_gyr, rot_raw) = sub.read(filename_gyr)
    filename_acc = f'input/{sensorname}_Accelerometer.csv'
    (t_acc, a) = sub.read(filename_acc)
    (t, rot_raw, a) = sub.synchronize(t_gyr, rot_raw, t_acc, a)
    (rot_vel, _, rot_abs) = conv.rotation(rot_raw, t, rot_mode='c',
                                          gyr_dict=gyr_dict)
    (v, t_step) = conv.velocity(a, t, acc_dict, rot_abs=rot_abs,
                                rot_vel=rot_vel)
    xyz = conv.xyz(t_step, v)

    omega = np.sqrt(rot_vel[:, 0]**2 + rot_vel[:, 1]**2 + rot_vel[:, 2]**2)
    E_rot = 0.4 * acc_dict['m'] * (acc_dict['r']**2) * omega**2
    E_trans = 0.5 * acc_dict['m'] * (v[:, 0]**2+v[:, 1]**2+v[:, 2]**2)
    E_kin = E_trans + E_rot

    if graph_dict['do_graph']:
        sub.graph2d(t, E_trans, typ='trans', filename=filename_acc,
                    string_check='A', graph_dict=graph_dict)
        sub.graph2d(t=t, y=E_rot, typ='rot', filename=filename_gyr,
                    string_check='G', graph_dict=graph_dict)
        sub.graph2d(t, E_kin, 'kin', graph_dict, filename_acc, 'A')
        if acc_dict['trajectory']:
            sub.graph3d(xyz=xyz, string_check='A', filename=filename_acc,
                        graph_dict=graph_dict)

    time_local_end = time.perf_counter()
    time_local = round((time_local_end - time_local_start), 3)
    print(f'took {time_local}s to process.')
    return (E_trans, E_rot, E_kin, t)


def failed(filename: str) -> None:
    """
    A function that is only there to say that there is no analysis method
    known.

    Parameters
    ----------
    filename : str
        The name of the file to be evaluated.
    """
    print(f'No analysis method is known for {filename}. Please check.')
    print('However, other files are still being analyzed.')


def str_gen(mmc_names: list, measurements: list,
            location: str = 'input/') -> list:
    """
    This function creates the input string.

    Parameters
    ----------
    mmc_names : list
        Names of the chips used.
    measurements : list
        What measurements have been made.
    location : str, optional
        Folder in which the data is stored. The default ist 'input/'

    Returns
    -------
    filenames : list
        List which contain the data strings
    """
    filenames = []
    for name in mmc_names:
        for measured in measurements:
            fstring = f'{location}{name}_{measured}.csv'
            filenames.append(fstring)

    return filenames
