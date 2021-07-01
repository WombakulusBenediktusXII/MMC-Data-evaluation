# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:08:55 2021
@author: Anton

Copyright (C) 2021  Smart Dust <contact@smartdust-dyt.de>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
import numpy as np

def velocity(a = np.array([]), t = np.array([]), err=float(0.001),
                  v_0 = np.array([0, 0, 0]), ing = bool(True)):
    '''
    velocity(a = np.array, t = np.array, v_0 = np.array([[0],[0],[0]]))

    The velocity function converts an acceleration into a velocity.
    The acceleration must be given in factors of g. All other parameters in SI basic units.
    It applies that the acceleration is constant in a time interval.

    Parameters
    ----------
    a : np.array. The change in acceleration in a time step
    t : np.array. The time intervals for which a was measured
    err : float. Value for error smoothing. By default 0.001
    v_0 : np.array. The initial velocity of the ball. By default (0, 0, 0)
    ing : boolean. Whether a is given in factors of g

    Returns
    -------
    v ; np. array. Current speed of the object

    '''
    err = 0.001
    if err == 0:
        err = 0.00001

    t = time_conv(t)
    if ing == True:
        g = 9.81
        a *= g # conversion to acceleration that does not depend on g

    ii = int(a.size/3)
    v = np.zeros([ii, 3])
    v[0,:] = v_0[:]
    for n in np.arange(1, (ii-1)): # the first data point is not required
        (a_err, _temp) = divmod(abs(a[n, :]),err)
        a_err *= err * np.sign(a[n, :])
        v[n, :] =  a_err[:] * t[:, n] - v[n-1, :]# calculates the new speed with the acceleration
#                                                  in the time step
    return v

def rotation(rot_raw=np.array([]), err=float(0.001), rot_0 = np.array([0, 0, 0])):
    '''
    rotation(rot_raw=np.array([]), rot_0 = np.array([0, 0, 0]))

    The rotation function smoothes the result of the rotation.

    Parameters
    ----------
    rot_raw : np.array. The measured rotation
    err : float. Value for error smoothing. By default 0.001
    rot_0 : np.array, optional. Starting rotation. By default (0, 0, 0)

    Returns
    -------
    rot : np.array. Smoothed rotation

    '''
    err = 0.001
    if err == 0:
        err = 0.00001

    ii = int(rot_raw.size/3)
    rot = np.zeros([ii, 3])
    rot[0, :] = rot_0[:]
    for n in np.arange(1, (ii-1)):
        (rot[n, :], _temp) = divmod(abs(rot_raw[n, :]), err)
        rot[n, :] *= err * np.sign(rot_raw[n, :])

    return rot


def time_conv(t = np.array([])):
    '''
    time_conv(t = np.array)

    The time_conv function adds up all the time steps, so that a timeline is created.

    Parameters
    ----------
    t : np.array. the measured time steps

    Returns
    -------
    t_step : np.array. the time of the individual time steps

    '''
    t_step = np.zeros([1, t.size])
    t_s = t[-1]-t[-2]
    for n in np.arange(1, t.size):
        t_step[0,n] = t_s

    return t_step

def string(_str=str, _filename=str, _temp=str):
    if _temp in 'G':
        _filename = _filename.replace("_Gyroscope.csv","").replace("input/","")

    elif _temp in 'A':
        _filename = _filename.replace("_Accelerometer.csv","").replace("input/","")

    else:
        pass

    return (_str + _filename + ',')
