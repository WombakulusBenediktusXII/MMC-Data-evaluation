# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:08:55 2021
@author: Anton
Version: v0.2-alpha

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

from datetime import date
import time
import numpy as np
from matplotlib import pyplot as plt
import processing as proces
import conversions as conv

def var():
    MMC_NAMES = ['Karl']  # Name of chips can be anything
    FILENAMES = ['input/Test_Datei.csv']  # Here you can enter the filenames
#                                           manuall, if DO_FILENAME_AUTO=True
#                                           then this will be ignored
    MEASUREMENTS = ['AccGyr']  # What has been measured. Change only if the
#                                method has been added or to delete an existing
#                                one.
#                                'Gyroscope', 'Accelerometer', 'AccGyr'
    M = 0.01496  # Mass of the balls with chip in kg
    R = 0.02925  # Radius of the spheres used in m
    DO_FILENAME_AUTO = True  # Whether the filenames should be created automatically
    DO_OUTPUT = False  # Automatically output .csv files with the calculated energy
    ACC_ERR = 0.001  # Value of error smoothing for accelerometer
    GYR_ERR = 0.01  # Value of error smoothing for gyroscoper
    SENSORPOS = np.array([1.2, 7.4, 4.5])  # Sensor position in 3d in mm.

    return (FILENAMES, M, R, DO_OUTPUT, ACC_ERR, GYR_ERR, MMC_NAMES,
            MEASUREMENTS, SENSORPOS, DO_FILENAME_AUTO)


def main():
    (FILENAMES, M, R, DO_OUTPUT, ACC_ERR, GYR_ERR,
     MMC_NAMES, MEASUREMENTS, SENSORPOS, DO_FILENAME_AUTO) = var()
    if DO_FILENAME_AUTO is True:
        FILENAMES = proces.str_gen(MMC_NAMES, MEASUREMENTS)
    MMC_LEN = len(MMC_NAMES)
    n = 0
    i = 0
    k = 0
    str_rot = 'Time in s, rotational energy in J, '
    str_trans = 'Time in s, translation energy in J, '
    str_kin = 'Time in s, kenetic energy in J, '
    for filename in FILENAMES:
        if 'Accelerometer' in filename:
            (E_trans_now, t) = proces.accelerometer(filename, M, ACC_ERR)
            if DO_OUTPUT is True:
                if n == 0:
                    E_trans = E_trans_now
                    n = 1
                    str_trans = conv.string(str_rot, filename, 'A')

                else:
                    E_trans = np.c_[E_trans, E_trans_now]
                    E_trans[:, 0] += E_trans_now[:]/MMC_LEN
                    str_trans = conv.string(str_rot, filename, 'A')

        elif 'Gyroscope' in filename:
            (E_rot_now, t) = proces.gyroscope(filename, M, R, GYR_ERR)
            if DO_OUTPUT is True:
                if i == 0:
                    E_rot = E_rot_now
                    i = 1
                    str_rot = conv.string(str_rot, filename, 'G')

                else:
                    E_rot = np.c_[E_rot, E_rot_now]
                    E_rot[:, 0] += E_rot_now[:]/MMC_LEN
                    str_rot = conv.string(str_rot, filename, 'G')
        elif 'AccGyr' in filename:
            (E_trans_now, E_rot_now,
             E_kin_now, t) = proces.accgyr(filename, M, R,
                                           accgry_err=(GYR_ERR, ACC_ERR),
                                           sensorpos=SENSORPOS)
            if DO_OUTPUT is True:
                if n == 0:
                    E_trans = E_trans_now
                    n = 1
                    str_trans = conv.string(str_rot, filename, 'AccGyr')
                else:
                    E_trans = np.c_[E_trans, E_trans_now]
                    E_trans[:, 0] += E_trans_now[:]/MMC_LEN
                    str_trans = conv.string(str_rot, filename, 'AccGyr')

                if i == 0:
                    E_rot = E_rot_now
                    i = 1
                    str_rot = conv.string(str_rot, filename, 'AccGyr')
                else:
                    E_rot = np.c_[E_rot, E_rot_now]
                    E_rot[:, 0] += E_rot_now[:]/MMC_LEN
                    str_rot = conv.string(str_rot, filename, 'AccGyr')

                if k == 0:
                    E_kin = E_kin_now
                    k = 1
                    str_kin = conv.string(str_kin, filename, 'AccGyr')
                else:
                    E_kin = np.c_[E_kin, E_kin_now]
                    E_kin[:, 0] += E_kin_now[:]/MMC_LEN
                    str_kin = conv.string(str_kin, filename, 'AccGyr')
        else:
            proces.failed(filename)

    if DO_OUTPUT is True:
        time_local_start = time.time()
        to_day = date.today()
        ID = np.random.randint(0, 10000)
        test_string = ''
        for string in MEASUREMENTS:
            test_string += string
        if 'Gyroscope' in test_string or 'AccGyr' in test_string:
            output_string = f'output/E_rot_{to_day}_id-{ID}.csv'
            np.savetxt(output_string, np.c_[t, E_rot], delimiter=",",
                       header=str_rot)
        if 'Accelerometer' in test_string or 'AccGyr' in test_string:
            output_string = f'output/E_trans_{to_day}_id-{ID}.csv'
            np.savetxt(output_string, np.c_[t, E_trans], delimiter=",",
                       header=str_trans)
        if 'AccGyr' in test_string:
            output_string = f'output/E_kin_{to_day}_id-{ID}.csv'
            np.savetxt(output_string, np.c_[t, E_kin], delimiter=",",
                       header=str_kin)
        time_local_end = time.time()
        time_local = round((time_local_end - time_local_start), 3)
        print(f'It took {time_local}s to create the output files.')
    return


if __name__ == '__main__':
    print('Program starts')
    time_global_start = time.time()
    main()
    time_global_end = time.time()
    time_global = round((time_global_end - time_global_start), 3)
    print(f'It took {time_global}s to run the program.')
    plt.show()
