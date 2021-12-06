# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 2021
@author: Anton
Version: v0.4-beta

In the conversions module, the data is converted into a different format.
These are:
    velocity(), rotation(), xyz(), timestep(), string(), intaxis(), rotvec(),
    _ktest()
"""

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline


def velocity(a: np.ndarray, t: np.ndarray, acc_dict: dict,
             **kwargs) -> (np.ndarray, np.ndarray):
    """
    Calculated the speed from the acceleration.

    Parameters
    ----------
    a : np.ndarray
        Acceleration vector.
    t : np.ndarray
        Continuous time.
    acc_dict : dict
        The dictionary which stores all constants for the accelerometer.
    **kwargs:
        rot_abs : np.ndarray, optional
            Absolute rotation if given, the velocity vector is output in the
            laboratory system.
        rot_vel : np.ndarray, optional
            Angular velocity if given, the angular acceleration is calculated
            out, provided that rot_abs is given.

    Raise
    -----
        RuntimeWaring
            If the absolute value of the maximum of a is less than 25*err

    Returns
    -------
    v : np.ndarray
        Velocity vector in the same format as a.
    t_step : np.ndarray
        Time steps
    """
    err = acc_dict['error']
    if acc_dict['in_g']:
        a *= 9.81

    (a_x, a_y) = a.shape
    v = np.zeros([a_x, 3])
    v[0, :] = acc_dict['start_velocity']
    t_step = timestep(t)
    if abs(a).max() <= err*25:
        raise RuntimeWarning('err is too large, the value is greater than the\
                             largest value.')

    elif err != 0 and err > 0:
        (a, _) = divmod(abs(a), err)
        a *= err * np.sign(a)

# This code may be incorrect: Start/
    if 'rot_abs' in kwargs:
        rot_abs = kwargs['rot_abs']
        (rot_abs_x, rot_abs_y) = rot_abs.shape
        if rot_abs_y == 3 and (rot_abs_x, rot_abs_y) == (a_x, a_y):
            a = rotvec(vec=a, rot=rot_abs)
            if 'rot_vel' in kwargs:
                rot_vel = kwargs['rot_vel']
                if rot_vel.shape == (rot_abs_x, rot_abs_y):
                    (x, y, z) = acc_dict['sensorpos']
                    for n in range(rot_abs_x):
                        rot_a = np.array([rot_vel[n, 1]*z - rot_vel[n, 2]*y,
                                          rot_vel[n, 2]*x - rot_vel[n, 0]*z,
                                          rot_vel[n, 0]*y - rot_vel[n, 1]*x])
                        a[n, :] -= rot_a / t_step[n]

            if acc_dict['g_interfered']:
                a[:, 2] -= 1.03*9.81
# /End
    elif acc_dict['g_interfered']:
        a -= 3.27  # = 9.81/3

    for n in range(1, a_x):
        v[n, :] = a[n, :] * t_step[n] - v[n-1, :]  # I can't explain the minus,
# but with a plus it always grows exponentially. And with the minus it
# corresponds to the expectations.
    v = intaxis(vec_1=t, vec_2=v, int_mode=acc_dict['integration_mode'],
                k=acc_dict['degree_of_spline'], s=acc_dict['smoothes'])
    return (v, t_step)


def rotation(rot_raw: np.ndarray, t: np.ndarray, rot_mode: str,
             gyr_dict: dict) -> (np.ndarray, np.ndarray):
    """
    Determines the rotation in the laboratory coordinate system from the
    measured rotation.
    Alternetive Mode: Determines the rotational speed in the laboratory
    coordinate system from the measured rotational speed.

    Parameters
    ----------
    rot_raw : np.ndarray
        Unsmoothed angular velocities.
    t : np.ndarray
        Continuous time.
    rot_mode : string
        Which mode to use.
            v : velocity mode -> angular velocity, t_step
            r : rotation mode -> absolute rotation, t_step
            c : combination mode -> angular velocity, t_step, absolute rotation
    gyr_dict : dict
         The dictionary which stores all constants.

    Raise
    -----
        RuntimeWaring
            If the absolute value of the maximum of rot_raw is less than 25*err
        ValusError
            If the rot_mode is not known.

    Returns
    -------
    rot_vel : np.array
        Angular velocity.
    t_step : np.array
        Time steps.
    rot_abs : np.array
        absolute rotation
    """
    err = gyr_dict['error']
    int_mode = gyr_dict['integration_mode']
    k = gyr_dict['degree_of_spline']
    s = gyr_dict['smoothes']
    rot_raw[0, :] = gyr_dict['start_rotation']
    (ii, _) = rot_raw.shape
    if abs(rot_raw).max() <= err*25:
        raise RuntimeWarning('err is too large, all results would be zero.')

    if gyr_dict['in_grad']:
        rot_raw *= np.pi/180

    if err != 0 and err > 0:
        rot_vel = abs(rot_raw) // err
        rot_vel *= err * np.sign(rot_raw)
    else:
        rot_vel = rot_raw

    t_step = timestep(t)
    if rot_mode in 'v':
        rot_vel = intaxis(vec_1=t, vec_2=rot_vel, int_mode=int_mode, k=k, s=s)
        rot_abs = None

    elif rot_mode in 'r':
        rot_abs = np.zeros(rot_vel.shape)
        for n in range(1, ii):
            rot_abs[n, :] = rot_abs[n-1, :] + rot_vel[n, :]*t_step[0, n]
        rot = intaxis(vec_1=t, vec_2=rot_abs, int_mode=int_mode, k=k, s=s)
        (_, rot_abs) = np.divmod(abs(rot), 2*np.pi) * np.sign(rot)
        rot_vel = None

    elif rot_mode in 'c':
        rot_vel = intaxis(vec_1=t, vec_2=rot_vel, int_mode=int_mode, k=k, s=s)
        rot_abs = np.zeros(rot_vel.shape)
        rot_abs[0, :] = rot_vel[0, :] * t_step[0]
        for n in range(1, ii):
            rot_abs[n, :] = rot_abs[n-1, :] + rot_vel[n, :]*t_step[n]
        (_, rot_abs) = np.divmod(abs(rot_abs), 2*np.pi) * np.sign(rot_abs)

    else:
        raise ValueError(f'The specified mode is not known: {rot_mode}.')

    return (rot_vel, t_step, rot_abs)


def xyz(t_step: np.ndarray, v: np.ndarray,
        xyz_0: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    """
    Calculates the trajectory from the velocity.

    Parameters
    ----------
    t_step : np.ndarray
        Time steps.
    v : np.ndarray
         Velocity vector.
    xyz_0 : np.ndarray, optional
        Start position of the sensor. The default is [0, 0, 0].

    Returns
    -------
    xyz_res : np.ndarray
        xyz position of the sensors during the measurement.
    """
    (lenx, leny) = v.shape
    xyz_res = np.zeros([lenx, leny])
    xyz_res[0, :] = xyz_0[:]
    for n in range(1, lenx):
        xyz_res[n, :] = v[n, :] * t_step[n] + xyz_res[n-1, :]

    return xyz_res


def timestep(t: np.ndarray) -> np.ndarray:
    """
    Determines dt for each measuring step.

    Parameters
    ----------
    t : np.ndarray
        Continuous time.

    Returns
    -------
    t_step : np.ndarray
        Time steps.
    """
    t_step = np.zeros(t.shape)
    t_step[:] = t[-1] / t.size
    return t_step


def intaxis(vec_1: np.ndarray, vec_2: np.ndarray, int_mode: str = 'i',
            k: int = 5, s: float = 0.8) -> np.ndarray:
    """
    Interpolate along all y-axes of the given array.

    Parameters
    ----------
    vec_1 : np.ndarray
        Values to be interpolated to.
    vec_2 : np.ndarray
        The array to be interpolated.
    int_mode : string
        Which mode to use.
            i : interpolation mode -> simple interpolation
            s : spline fit mode -> 1-D smoothing spline fit
            a : average mode -> average with value and +-k values
    k : int, optional
        Degree of the smoothing spline or points for averaging in both
        directions.
        The default is int(5).
    s : float, optional
        Positive smoothing factor. The default is 0.8.

    Raise
    -----
    TypeError
        If k is not a integer.
    ValusError
        If the int_mode is not known.

    Returns
    -------
    vec_res : np.ndarray
        Interpolated vector.
    """
    if type(k) is not int:
        raise TypeError(f'k has the wrong type. Is {type(k)} and not int.')

    (x, y) = vec_2.shape
    vec_res = np.zeros((x, y))
    if int_mode in ['s', 'S']:
        for n in range(0, y):
            spl = UnivariateSpline(x=vec_1, y=vec_2[:, n], k=k, s=s,
                                   check_finite=False)
            vec_res[:, n] = spl(vec_1)

    elif int_mode in ['i', 'I']:
        k = _ktest(k)
        for n in range(0, y):
            fun = interp1d(x=vec_1, y=vec_2[:, n], kind=k,
                           fill_value='extrapolate', assume_sorted=True)
            vec_res[:, n] = fun(vec_1)

    elif int_mode in ['a', 'A']:
        _vec_2 = vec_2
        for n in range(0, x):
            n_low = n - k
            n_high = n + k
            n_low = max(n_low, 0)
            n_high = min(n_high, x)

            for m in range(0, y):
                _vecm = _vec_2[n_low:n_high, m]
                vec_res[n, m] = _vecm.mean()

    else:
        raise ValueError(f'The specified mode is not known: {int_mode}.')

    return vec_res


def string(str_: str, filename: str, string_check: str) -> str:
    """
    Deletes everything from the string except the name of the
    MMC and adds that to the given string.

    Parameters
    ----------
    str_ : str
        The string to be appended to.
    filename : str
        The string to be processed.
    string_check : str
        What the string to be processed denotes.
        Character :  Meaning
           'A'    :  Accelerometer
           'AL'   :  AmbientLight
           'G'    :  Gyroscope
           'GR'   :  Gravity
           'LA'   :  LinearAcceleration
           'M'    :  Magnetometer
           'P'    :  Pressure
           'Q'    :  Quaterion
           'T'    :  Temperature

    Raise
    -----
         ValusError
             If the string_check is not known.

    Returns
    -------
    _str : str
        The edited string.
    """
    if string_check in 'G':
        filename = filename.replace('_Gyroscope.csv', '').replace('input/', '')
    elif string_check in 'A':
        filename = filename.replace('_Accelerometer.csv', '').replace('input/','')
    elif string_check in ['La', 'LA']:
        filename = filename.replace('_LinearAcceleration.csv', '').replace('input/', '')
    elif string_check in 'Q':
        filename = filename.replace('_Quaterion.csv', '').replace('input/', '')
    elif string_check in 'M':
        filename = filename.replace('_Magnetometer.csv', '').replace('input/', '')
    elif string_check in ['Gr', 'GR']:
        filename = filename.replace('_Gravity.csv', '').replace('input/', '')
    elif string_check in ['Al', 'AL']:
        filename = filename.replace('_AmbientLight.csv', '').replace('input/', '')
    elif string_check in 'P':
        filename = filename.replace('_Pressure.csv', '').replace('input/', '')
    elif string_check in 'T':
        filename = filename.replace('_Temperature.csv', '').replace('input/', '')
    elif string_check in 'AccGyr':
        filename = filename.replace('_AccGyr.csv', '').replace('input/', '')
        filename += ' AccGyr'
    else:
        raise ValueError(f'{string_check} is not a valid input.')

    return (str_ + filename + ',')


def rotvec(vec: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Calculates the x, y, z components of vec when it is rotated by rot. As many
    1d vectors as desired can be rotated.

    Parameters
    ----------
    vec : np.ndarray
        The vector to be rotated.
    rot : np.ndarray
        Angles for the rotation matrix.

    Raise
    -----
    ValueError
        If vec and rot do not have the same number of elements on the
        x-axis.
    ValueError
        If rot does not have three elements on the y-axis.

    Returns
    -------
    vec_parts : np.array
        x, y, z-parts of the rotated vec.
    """
    (vec_x, _) = vec.shape
    (rot_x, rot_y) = rot.shape
    if vec_x != rot_x:
        raise ValueError(f'rot and vec do not have the same length on the x-axis: {rot_x} != {vec_x}')
    elif rot_y != 3:
        raise ValueError(f'rot does not have three elements on the y-axis. y = {rot_y}')

    for n in range(0, vec_x):
        vec[n, :] = np.sqrt(vec[n, 0]**2 + vec[n, 1]**2 + vec[n, 2]**2)
        vec[n, 0] *= np.sin(rot[n, 2]) * np.cos(rot[n, 0])
        vec[n, 1] *= np.sin(rot[n, 2]) * np.sin(rot[n, 0])
        vec[n, 2] *= np.cos(rot[n, 2])

    return vec


def _ktest(k: int = 3) -> str:
    """
    Converts the input into a string that can be processed by
    scipy.interpolate.interp1d.

    Parameters
    ----------
    k : int
        what is to be tested.

    Raise
    -----
    TypeError
        If k is not a integer.
    ValueError
        If k is not between 0 and 3.

    Returns
    -------
    res : string
        String that can be processed by scipy.interpolate.interp1d.
    """
    if type(k) is not int:
        raise TypeError(f'k is not a int. k = {type(k)}')
    if k == 0:
        res = 'zero'
    elif k == 1:
        res = 'slinear'
    elif k == 2:
        res = 'quadratic'
    elif k == 3:
        res = 'cubic'
    else:
        raise ValueError(f'k is not a valid argument. k can be between 0...3. k={k}')

    return res
