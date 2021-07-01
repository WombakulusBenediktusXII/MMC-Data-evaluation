# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:52:07 2021
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
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time
import conversions as conv

def accelerometer(filename=str(''), m=float(1), err=float(0.001)):
    time_local_start = time.time()
    data_LinAcc = read(filename)
    t = data_LinAcc[:, 2]
    a = data_LinAcc[:, 3:]
    v = conv.velocity(a, t, err)
    m /= 1000
    E_trans = 0.5 * m * (v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2)

    _stringf = FormatStrFormatter('%1.3e')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Time in s')
    ax.set_ylabel('$E_{trans}$ in J')
    ax.set_ylim(E_trans.min(), E_trans.max())
    ax.set_title('translation energy as a function of time')
    ax.yaxis.set_major_formatter(_stringf)
    ax.plot(t, E_trans)
    ax.grid()
    time_local_end = time.time()
    time_local = round((time_local_end - time_local_start), 3)
    print(f'It took {time_local}s to process {filename}.')
    return (E_trans, t)

def gyroscope(filename=str(''), m=float(1), r=float(1), err=float(0.001)):
    time_local_start = time.time()
    data_Gyr = read(filename)
    t = data_Gyr[:, 2]
    rot_raw = data_Gyr[:, 3:]
    rot = conv.rotation(rot_raw, err)
    omega = (rot[:, 0]**2 + rot[:, 1]**2 + rot[:, 2]**2)
    m /= 1000
    r /= 1000
    E_rot = 0.2 * m * r**2 * omega[:]  # This is for a homogeneous solid sphere
#                                     so it's just a very simple approximation.

    _stringf = FormatStrFormatter('%1.3e')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Time in s')
    ax.set_ylabel('$E_{rot}$ in J')
    ax.set_ylim(E_rot.min(), E_rot.max())
    ax.yaxis.set_major_formatter(_stringf)
    ax.set_title('rotational energy as a function of time')
    ax.plot(t, E_rot)
    ax.grid()
    time_local_end = time.time()
    time_local = round((time_local_end - time_local_start), 3)
    print(f'It took {time_local}s to process {filename}.')
    return (E_rot, t)

def failed(filename=str('')):
    return print(f'No analysis method is known for {filename}. Please check.')

def read(filename=str('')):
    return np.genfromtxt(filename, delimiter=',', skip_header=1)

def sumforline(filename=str('')):
    with open(filename) as f:
        return sum(1 for line in f)
