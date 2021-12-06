# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 2021
@author: Anton
Version: v0.4-beta

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
__version__ = 'v0.4-beta'
__author__ = 'SmartDust'

import time
import os
from multiprocessing import Pool, Manager
from matplotlib import pyplot as plt

import processing as proces
import subprocessing as sub
from config_parser import get_config
from data_output import data_storer


def main() -> None:
    '''
    This is the main function of this programm
    '''
    global time_global_start
    (main_dict, acc_dict,
     gyr_dict, graph_dict) = get_config(filename='config.ini')

    if main_dict['filenames_auto']:
        filenames = proces.str_gen(main_dict['names'],
                                   main_dict['measurements'])
    else:
        filenames = main_dict['filenames']
    MMC_LEN = len(filenames)
    data = []

#  Check if the graph storage folder exists, if not it will be created.
    if graph_dict['save_graph'] or main_dict['multi_processing']:
        if not os.path.exists('saved_graphs'):
            os.makedirs('saved_graphs')
            print('The directory "saved_graph" has been created automatically.\
                  \nIn these will be the saved graphs.\n')

#  Decision whether multiprocessing should be applied
    mmc_lens = MMC_LEN + 2*str(filenames).count('AccGyr')
    if main_dict['multi_processing'] is None:
        if mmc_lens > 8:
            main_dict['multi_processing'] = True
            print('It has been decided to use multiprocessing.')

        else:
            main_dict['multi_processing'] = False
            print('It has been decided not to use multiprocessing.\n')

#  Ask if multiprocessing should be used in case of unoptimal conditions.
    elif main_dict['multi_processing'] and mmc_lens <= 8:
        print(f'Multiprocessing should be applied, but only {mmc_lens} file(s)\
have been specified. This can lead to slower processing.')
        time_mid_point = time.perf_counter()
        question = 'Should multiprocerssing nevertheless be continued?'
        main_dict['multi_processing'] = sub.input_test(question)
        time_global_start += time.perf_counter() - time_mid_point

#  Inisalization and application of multiprocessing.
    if main_dict['multi_processing']:
        graph_dict['save_graph'] = True
        graph_dict['do_graph'] = True
        workers = main_dict['max_processes']
        filenames.sort(key=sub.filename_sorting_key)
        print(f'Multiprocessing has been started. {workers} simultaneous processes are used.')
        if mmc_lens > 32:
            manager = Manager()
            acc_dict = manager.dict(acc_dict)
            gyr_dict = manager.dict(gyr_dict)
            graph_dict = manager.dict(graph_dict)
            print(f'The dictionaries are shared between all processes to save\
 RAM. Reason {mmc_lens} files need to be analyzed.')

        iterable = [(filenames[n], acc_dict, gyr_dict, graph_dict)
                    for n in range(MMC_LEN)]
        with Pool(processes=workers) as pool:
            pools = pool.starmap_async(proces.main, iterable)
            data = pools.get()

#  Serial processing of the data.
    else:
        for filename in filenames:
            data_now = proces.main(filename, acc_dict, gyr_dict, graph_dict)
            data.append(data_now)

    if main_dict['save_output']:
        time_local_start = time.perf_counter()
        data_storer(data, main_dict['save_formatter'])
        time_local_end = time.perf_counter()
        time_local = round((time_local_end - time_local_start), 3)
        print(f'It took {time_local}s to create the output files.')


if __name__ == '__main__':
    print('Program starts...\n')
    time_global_start = time.perf_counter()
    main()
    time_global = round((time.perf_counter() - time_global_start), 3)
    print(f'\nIt took {time_global}s to run the program.')
    plt.show()
