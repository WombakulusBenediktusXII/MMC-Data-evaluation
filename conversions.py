# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:08:55 2021
@author: Anton
Version: v0.2-alpha

In the conversions module, the data is converted into a different format.
These are:
    velocity(), rotation(), xyz(), timestep(), string(), intaxis(), rotvec(),
    _ktest()
"""

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline


def velocity(a=np.array([]), t=np.array([]), err=0.1, v_0=np.array([0, 0, 0]),
             in_g=True, out_g=True, rot_abs=np.array([]), rot_vel=np.array([]),
             sensorpos=np.array([1.2, 7.4, 4.5]), int_mode='a', k=50, s=0.8):
    """
    Calculated the speed from the acceleration.

    Parameters
    ----------
    a : np.array, mandatory
        Acceleration vector.
    t : np.array, mandatory
        Continuous time.
    err : float, optional
        Value of error smoothing. The default is 0.1.
    v_0 : np.array, optional
        np.array. The default is np.array([0, 0, 0]).
    in_g : boolean, optional
        Whether the acceleration is given in factors of g. The default is True.
    out_g : boolean, optional
        Whether the gravitation has interfered with the sensors.
        The default is True.
    rot_abs : np.array, optional
        Absolute rotation if given, the velocity vector is output in the
        laboratory system.
    rot_vel : np.array, optional
        Angular velocity if given, the angular acceleration is calculated out,
        provided that rot_abs is given.
    sensorpos : np.array, optional
        Sensor position in 3d in mm. The default is np.array([1.2,7.4,4.5])
    int_mode : string, optional
        Which int_mode to use. See conversions.intaxis. The default is 'a'
    k : int, optional
        Degree of the smoothing spline or points for averaging in both
        directions.
        The default is 5.
    s : float, optional
        Positive smoothing factor. The default is 0.8.

    Returns
    -------
    v : np.array
        Velocity vector in the same format as a.
    t_step : np.array
        Time steps
    """
    if in_g is True:
        a *= 9.81

    (ii, _) = a.shape
    v = np.zeros([ii, 3])
    v[0, :] = v_0
    t_step = timestep(t)
    if abs(a).max() <= err*25:
        raise RuntimeWarning('err is too large, the value is greater than the largest value.')
    elif err != 0 and err > 0:
        (a_err, _) = divmod(abs(a), err)
        a_err *= err * np.sign(a)
        a = a_err
    # This code may be incorrect: Start/
    if rot_abs.shape != (0,) and rot_abs.shape == a.shape:
        a = rotvec(vec=a, rot=rot_abs)
        if rot_vel.shape != (0,) and rot_vel.shape == a.shape:
            (x, y, z) = sensorpos / 1000
            for n in np.arange(0, ):
                a[n, :] -= np.array([[rot_vel[n, 1]*z - rot_vel[n, 2]*y],
                                [rot_vel[n, 2]*x - rot_vel[n, 0]*z],
                                [rot_vel[n, 0]*y - rot_vel[n, 1]*x]])/t_step[n]

        if out_g is True:
            a[:, 2] -= 1.03*9.81
    # /End
    elif out_g is True:
        a -= (1.03*9.81)/3
    for n in np.arange(1, ii):
        v[n, :] = a[n, :] * t_step[n] - v[n-1, :]  # I can't explain the minus,
# but with a plus it always grows exponentially. And with the minus it corresponds to the expectations.
    v = intaxis(t=t, vec=v, int_mode=int_mode, k=k, s=s)
    return (v, t_step)


def rotation(rot_raw=np.array([]), t=np.array([]), rot_mode='', err=0.1,
             rot_0=np.array([0, 0, 0]), in_grad=True, int_mode='s', k=5, s=0.8):
    """
    Determines the rotation in the laboratory coordinate system from the
    measured rotation.
    Alternetive Mode: Determines the rotational speed in the laboratory
    coordinate system from the measured rotational speed.

    Parameters
    ----------
    rot_raw : np.array, mandatory
        Unsmoothed angular velocities with [[x_1, y_1, z_1],...,[x_n,y_n,z_n]].
    t : np.array, mandatory
        Continuous time.
    rot_mode : string, mandatory
        Which mode to use.
            v : velocity mode -> angular velocity, t_step
            r : rotation mode -> absolute rotation, t_step
            c : combination mode -> angular velocity, t_step, absolute rotation
    err : float, optional
        Value of error smoothing. The default is 0.1.
    rot_0 : np.array, optional
        Initial rotation. The default is np.array([0, 0, 0]).
        In the same unit as rot_raw.
    in_grad :  boolean, optional
        Whether rot_raw is specified in °/s if true than conversion in rad/s.
        The default ist True.
    int_mode : string, optional
        Which int_mode to use. See conversions.intaxis. The default is 's'
    k : int, optional
        Degree of the smoothing spline or points for averaging in both directions.
        The default is 5.
    s : float, optional
        Positive smoothing factor. The default is 0.8.

    Returns
    -------
    rot : np.array
        Angular velocity or absolute rotation.
    t_step : np.array
        Time steps.
    rot_abs : np.array
        absolute rotation
    """
    rot_raw[0, :] = rot_0[:]
    (ii, _) = rot_raw.shape
    if abs(rot_raw).max() <= err*25:
        raise RuntimeWarning('err is too large, all results would be zero.')
    if in_grad is True:
        rot_raw *= np.pi/180

    if err != 0 and err > 0:
        rot_vel = abs(rot_raw) // err
        rot_vel *= err * np.sign(rot_raw)
    else:
        rot_vel = rot_raw


    t_step = timestep(t)
    if rot_mode in 'v':
        rot = intaxis(t=t, vec=rot_vel, int_mode=int_mode, k=k, s=s)

    elif rot_mode in 'r':
        rot_abs = np.zeros(rot_vel.shape)
        for n in np.arange(1, ii):
            rot_abs[n, :] = rot_abs[n-1, :] + rot_vel[n, :]*t_step[0, n]
        rot = intaxis(t=t, vec=rot_abs, int_mode=int_mode, k=k, s=s)
        (_, rot) = np.divmod(abs(rot), 2*np.pi) * np.sign(rot)

    elif rot_mode in 'c':
        rot_vel = intaxis(t=t, vec=rot_vel, int_mode=int_mode, k=k, s=s)
        rot_abs = np.zeros(rot_vel.shape)
        rot_abs[0, :] = rot_vel[0, :] * t_step[0]
        for n in np.arange(1, ii):
            rot_abs[n, :] = rot_abs[n-1, :] + rot_vel[n, :]*t_step[n]
        (_, rot_abs) = np.divmod(abs(rot_abs), 2*np.pi) * np.sign(rot_abs)
        return (rot_vel, t_step, rot_abs)

    else:
        raise ValueError(f'The specified mode is not known: {mode}.')

    return (rot, t_step)


def xyz(t_step=np.array([]), v=np.array([]), xyz_0=np.array([0, 0, 0])):
    """
    Calculates the trajectory from the velocity.

    Parameters
    ----------
    t_step : np.array, mandatory
        Time steps.
    v : np.array, mandatory
         Velocity vector.
    xyz_0 : np.array, optional
        Start position of the sensor. The default is [0, 0, 0].

    Returns
    -------
    xyz : np.array
        xyz position of the sensors during the measurement.
    """
    (lenx, leny) = v.shape
    xyz = np.zeros([lenx, leny])
    xyz[0, :] = xyz_0[:]
    for n in np.arange(1, lenx):
        xyz[n, :] = v[n, :] * t_step[n] + xyz[n-1, :]

    return xyz


def timestep(t=np.array([])):
    """
    Determines dt for each measuring step.

    Parameters
    ----------
    t : np.array, mandatory
        Continuous time.

    Returns
    -------
    t_step : np.array
        Time steps.
    """
    t_step = np.zeros(t.shape)
    t_step[:] = t[-1] / t.size
    return t_step


def intaxis(t=np.array([]), vec=np.array([]), int_mode='', k=5, s=0.8):
    """
    Interpolate along all y-axes of the given array.

    Parameters
    ----------
    t : np.array, mandatory
        Continuous time.
    vec : np.array, mandatory
        The array to be interpolated.
    int_mode : string, mandatory
        Which mode to use.
            i : interpolation mode -> simple interpolation
            s : spline fit mode -> 1-D smoothing spline fit
            a : average mode -> average with value and +-k values
    k : int, optional
        Degree of the smoothing spline or points for averaging in both directions.
        The default is int(5).
    s : float, optional
        Positive smoothing factor. The default is 0.8.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    vec : TYPE
        DESCRIPTION.
    """
    if type(k) is not int:
        raise TypeError(f'k has the wrong type. Is {type(k)} and not int.')
    (x, y) = vec.shape
    if int_mode in 's':
        for n in np.arange(y):
            spl = UnivariateSpline(x=t, y=vec[:, n], k=k, s=s,
                                   check_finite=False)
            vec[:, n] = spl(t)
    elif int_mode in 'i':
        k = _ktest(k)
        for n in np.arange(y):
            fun = interp1d(x=t, y=vec[:, n], kind=k, fill_value='extrapolate',
                           assume_sorted=True)
            vec[:, n] = fun(t)
    elif int_mode in 'a':
        _vec = vec
        for n in np.arange(x):
            n_low = n - k
            n_high = n + k
            if n_low <= 0:
                n_low = 0
            if n_high >= x:
                n_high = x

            for m in np.arange(y):
                _vecm = _vec[n_low:n_high, m]
                vec[n, m] = _vecm.mean()
    else:
        raise ValueError(f'The specified mode is not known: {int_mode}.')

    return vec


def string(str_='', filename='', string_check=''):
    """
    Deletes everything from the string except the name of the
    MMC and adds that to the given string.

    Parameters
    ----------
    str_ : str, mandatory
        The string to be appended to.
    filename : str, mandatory
        The string to be processed.
    string_check : str, mandatory
        What the string to be processed denotes.
        Character :  Meaning
           'A'    :  Accelerometer
           'AL'   :  AmbientLight
           'G'    :  Gyroscope
           'Gr'   :  Gravity
           'LA'   :  LinearAcceleration
           'M'    :  Magnetometer
           'P'    :  Pressure
           'Q'    :  Quaterion
           'T'    :  Temperature

    Returns
    -------
    str_ : str
        The edited string.
    """
    if string_check in 'G':
        filename = filename.replace('_Gyroscope.csv', '').replace('input/', '')
    elif string_check in 'A':
        filename = filename.replace('_Accelerometer.csv', '').replace('input/', '')
    elif string_check in 'LA':
        filename = filename.replace('_LinearAcceleration.csv', '').replace('input/', '')
    elif string_check in 'Q':
        filename = filename.replace('_Quaterion.csv', '').replace('input/', '')
    elif string_check in 'M':
        filename = filename.replace('_Magnetometer.csv', '').replace('input/', '')
    elif string_check in 'Gr':
        filename = filename.replace('_Gravity.csv', '').replace('input/', '')
    elif string_check in 'AL':
        filename = filename.replace('_AmbientLight.csv', '').replace('input/', '')
    elif string_check in 'P':
        filename = filename.replace('_Pressure.csv', '').replace('input/', '')
    elif string_check in 'T':
        filename = filename.replace('_Temperature.csv', '').replace('input/', '')
    elif string_check in 'AccGyr':
        filename = filename.replace('_AccGyr.csv', '').replace('input/', '')
    else:
        raise ValueError(f'{string_check} is not a valid input.')
    string = str_ + filename + ','
    return string


def rotvec(vec=np.array([]), rot=np.array([])):
    """
    Calculates the x, y, z components of vec when it is rotated by rot. As many vectors
    as desired can be rotated. But vec.size = rot.size.

    Parameters
    ----------
    vec : np.array, mandatory
        The vector to be rotated.
    rot : np.array, mandatory
        Angles for the rotation matrix.

    Returns
    -------
    vec_parts : np.array
        x, y, z-parts of the rotated vec.
    """
    if vec.shape != rot.shape:
        raise ValueError(f'rot and vec have not the same shape: {rot.shape} != {vec.shape}')

    (ii, _) = vec.shape
    for n in np.arange(0, ii):
        vec[n, :] = np.sqrt(vec[n, 0]**2 + vec[n, 1]**2 + vec[n, 2]**2)
        vec[n, 0] *= np.sin(rot[n, 2]) * np.cos(rot[n, 0])
        vec[n, 1] *= np.sin(rot[n, 2]) * np.sin(rot[n, 0])
        vec[n, 2] *= np.cos(rot[n, 2])

    return vec


def _ktest(k=float(3)):
    """
    Converts the input into a string that can be processed by
    scipy.interpolate.interp1d.

    Parameters
    ----------
    k : float, mandatory
        what is to be tested.

    Returns
    -------
    string : string
        String that can be processed by scipy.interpolate.interp1d.
    """
    if k == 0:
        return 'zero'
    elif k == 1:
        return 'slinear'
    elif k == 2:
        return 'quadratic'
    elif k == 3:
        return 'cubic'
    else:
        raise ValueError(f'k is not a valid argument. k can be between 0 and 3. k={k}')
