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
import processing as proces
import conversions as conv
from matplotlib import pyplot as plt
import time
import datetime

def var():
    MMC_names = ['Hans']#, 'Franz', 'Karl', 'Otto', 'Peter']  # Name of chips can be anything
    filenames = ['input/Test_Datei.csv']  # Here you can enter the filenames manuall if auto=True then
    #                                       this will be ignored
    measurements = ['Gyroscope', 'Accelerometer']  # What has been measured. Change only if the method
    #                                                has been added or to delete an existing one.
    m = 1  # Mass of the balls with chip in g
    r = 1  # Radius of the spheres used in mm
    auto = True  # Whether the filenames should be created automatically
    do_output = False  # Automatically output .csv files with the calculated energy.
    # Nutz bitte ermal do_output = True, nur wenn ihr eine datei verwendet z.B. ihr sicher seit das
    # alle Dateinen von einen measurments element gleich lang sind. Das ist nicht gegeben da noch keine
    # Synronisaton der Messdaten statt findet.
    err_Acc = 0.1
    err_Gyr = 0.1
    return (MMC_names,filenames,measurements,m,r,auto,do_output,err_Acc,err_Gyr)

def main():
    (MMC_names,filenames,measurements,m,r,auto,do_output,err_Acc,err_Gyr) = var()
    n = 0
    i = 0
    E_trans = np.empty([])
    E_rot = np.empty([])
    str_rot = 'Time in s, rotational energy in J, '
    str_trans = 'Time in s, rotational energy in J, '
    if auto is True:
        filenames = []
        for name in MMC_names:
            for measured in measurements:
                fstring = f'input/{name}_{measured}.csv'
                filenames.append(fstring)

    for filename in filenames:
        if 'Accelerometer' in filename:
            (E_trans_now, t_trans) = proces.accelerometer(filename, m, err_Acc)
            if do_output is True:
                if n == 0:
                    E_trans = E_trans_now
                    n = 1
                    str_trans = conv.string(str_rot, filename, 'A')

                else:
                    E_trans = np.c_[E_trans, E_trans_now]
                    E_trans[:, 0] += E_trans_now[:]/len(MMC_names)
                    str_trans = conv.string(str_rot, filename, 'A')

        elif 'Gyroscope' in filename:
            (E_rot_now, t_rot) = proces.gyroscope(filename, m, r, err_Gyr)
            if do_output is True:
                if i == 0:
                    E_rot = E_rot_now
                    i = 1
                    str_rot = conv.string(str_rot, filename, 'G')

                else:
                    E_rot = np.c_[E_rot, E_rot_now]
                    E_rot[:, 0] += E_rot_now[:]/len(MMC_names)
                    str_rot = conv.string(str_rot, filename, 'G')

        else:
            proces.failed(filename)

    if do_output is True:
        time_local_start = time.time()
        _date = datetime.date.today()
        _rng = np.random.randint(0, 100)
        _str = f'output/E_rot_{_date}_{_rng}.csv'
        np.savetxt(_str, np.c_[t_rot, E_rot], delimiter=",", header=str_rot)
        _str = f'output/E_trans_{_date}_{_rng}.csv'
        np.savetxt(_str, np.c_[t_trans, E_trans], delimiter=",", header=str_trans)
        time_local_end = time.time()
        time_local = round((time_local_end - time_local_start), 3)
        print(f'It took {time_local}s to create the output files.')
    return

if __name__ == '__main__':
    time_global_start = time.time()
    main()
    time_global_end = time.time()
    time_global = round((time_global_end - time_global_start), 3)
    print(f'It took {time_global}s time.')
    plt.show()