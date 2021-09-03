# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:52:07 2021
@author: Anton
Version: v0.2-alpha

The Processing module contains all functions responsible for the direct processing
of raw data.
These are:
    accelerometer(), gyroscope(), accgyr(), failed(), str_gen()
"""

import numpy as np
import time
import subprocessing as sub
import conversions as conv


def accelerometer(filename='', M=1, ACC_ERR=0.001, trajectory=False):
    """
    Read the file of the accelerometer and calculate the speed from it.
    From this the translations energy is calculated and output as a
    diagram over time.

    Parameters
    ----------
    filename : str, mandatory
        The name of the file to be evaluated.
    M : float in kg, optional
        Mass of the object. The default is 1.
    ACC_ERR : float, optional
        Value of error smoothing. The default is 0.001.
    trajectory: bool, optional
        Whether a trajectory should be created. The default is False.

    Returns
    -------
    E_trans : np.array
        Translational energy.
    t : np.array
        Time of measurement for the individual translation energies.
    """
    time_local_start = time.time()
    (t, a) = sub.read(filename)
    (v, t_step) = conv.velocity(a, t, ACC_ERR)
    E_trans = 0.5 * M * (v[:, 0]**2+v[:, 1]**2+v[:, 2]**2)
    sub.graph2d(t, E_trans, typ='trans', filename=filename, string_check='A')
    if trajectory is True:
        xyz = conv.xyz(t_step, v)
        sub.graph3d(xyz=xyz, filename=filename, density=int(1000))
    time_local_end = time.time()
    time_local = round((time_local_end - time_local_start), 3)
    print(f'It took {time_local}s to process {filename}.')
    return (E_trans, t)


def gyroscope(filename='', M=1, R=1, GYR_ERR=0.1):
    """
    Reads the file of the gyroscope. The rotation energy is then calculated from this.
    BUT it is calculated for a solid full sphere, so for everything but a sphere very
    poorly suited. The rotational energy is output as a graph over time.

    Parameters
    ----------
    filename : str, mandatory
        The name of the file to be evaluated.
    M : float, optional
        Mass of the object in kg. The default is 1.
    R : float, optional
        radius of the object in m. The default is 1.
    err : TYPE, optional
        Value of error smoothing. The default is 0.1.

    Returns
    -------
    E_rot : np.array
        Rotational energy.
    t : np.array
        Time of measurement for the individual translation energies.
    """
    # ist noch nicht an das neue sub.read angepasst
    # ist noch nicht an die winkel angepasst
    time_local_start = time.time()
    (t, rot_raw) = sub.read(filename)
    (rot, t_step) = conv.rotation(rot_raw, t, rot_mode='v', err=GYR_ERR)
 #  This is for a homogeneous solid sphere so it's just a very simple approximation.
    omega = np.sqrt(rot[:, 0]**2 + rot[:, 1]**2 + rot[:, 2]**2)
    E_rot = 0.4 * M * (R**2) * omega**2
    sub.graph2d(t=t, y=E_rot, typ='rot', filename=filename, string_check='G')
    time_local_end = time.time()
    time_local = round((time_local_end - time_local_start), 3)
    print(f'It took {time_local}s to process {filename}.')
    return (E_rot, t)


# A better method must be found to synchronize the two measurement series.
def accgyr(filename='',  M=1, R=1, accgry_err=(0.01, 0.1),
           sensorpos=np.array([1.2, 7.4, 4.5])):
    """
    Reads the data from the gyroscope and the accelerometer. Calculated the rotational
    energy and the translational energy and subtracted the angular acceleration from
    the measured acceleration.

    Parameters
    ----------
    filename : str, mandatory
        DESCRIPTION. The default is str().
    M : float, optional
        Mass of the object in kg. The default is 1.
    R : float, optional
        radius of the object in m. The default is 1.
    accgry_err : tuple, optional
        Value of error smoothing. First value is for the gyroscope,
        the second for the accelerometer. The default is (0.01, 0.1).
    sensorpos : np.array, optional
        Sensor position in 3d in mm. The default is [1.2, 7.4, 4.5].

    Returns
    -------
    E_trans : np.array
        Translational energy.
    E_rot : np.array
        Rotational energy.
    E_kin : np.array
        Kenetic energy
    t : np.array
        Time of measurement for the individual energies.
    """
    time_local_start = time.time()
    sensorname = filename.replace("_AccGyr.csv", "").replace("input/", "")
    filename_gyr = f'input/{sensorname}_Gyroscope.csv'
    (t_gyr, rot_raw) = sub.read(filename_gyr)
    filename_acc = f'input/{sensorname}_Accelerometer.csv'
    (t_acc, a) = sub.read(filename_acc)
    (t, rot_raw, a) = sub.synchronize(t_gyr, rot_raw, t_acc, a)
    (rot_vel, _, rot_abs) = conv.rotation(rot_raw, t, rot_mode='c',
                                          err=accgry_err[0])
    (v, t_step) = conv.velocity(a, t, rot_abs=rot_abs, rot_vel=rot_vel,
                                sensorpos=sensorpos, err=accgry_err[1],
                                in_g=False)
    xyz = conv.xyz(t_step, v)

#  This is for a homogeneous solid sphere so it's just a very simple approximation.
    omega = np.sqrt(rot_vel[:, 0]**2 + rot_vel[:, 1]**2 + rot_vel[:, 2]**2)
    E_rot = 0.4 * M * (R**2) * (omega**2)

    v_abs = np.sqrt(v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2)
    E_trans = 0.5 * M * (v_abs**2)

    E_kin = E_trans + E_rot

    sub.graph2d(t, E_trans, typ='trans', filename=filename_acc,
                string_check='A')
    sub.graph2d(t, E_rot, typ='rot', filename=filename_gyr, string_check='G')
    sub.graph3d(xyz, rot_abs, filename=filename_acc, string_check='A')
    time_local_end = time.time()
    time_local = round((time_local_end - time_local_start), 3)
    print(f'It took {time_local}s to process from {sensorname} the gyroscope and accelerometer.')
    return (E_trans, E_rot, E_kin, t)


def failed(filename=''):
    """
    A function that is only there to say that there is no analysis method known.

    Parameters
    ----------
    filename : str, mandatory
        The name of the file to be evaluated.
    """
    return print(f'No analysis method is known for {filename}. Please check.')


def str_gen(MMC_names=[], measurements=[], location='input/'):
    """
    This function creates the input string.

    Parameters
    ----------
    MMC_names : list, mandatory
        Names of the chips used.
    measurements : list, mandatory
        What measurements have been made.
    location : str, optional
        Folder in which the data is stored. The default ist 'input/'

    Returns
    -------
    filenames : list
        List which contain the data strings
    """
    filenames = []
    for name in MMC_names:
        for measured in measurements:
            fstring = f'{location}{name}_{measured}.csv'
            filenames.append(fstring)
    return filenames
