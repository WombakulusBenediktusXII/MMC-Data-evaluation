# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 2021
@author: Anton
Version: v0.4-beta

The output_data module stores the calculated data in external files.
These are:
    data_storer(), save_to_file(), output_array(), data_array_test(),
    data_array_first(), data_array_add(), str_edit()
"""


import os
from datetime import datetime
import numpy as np

import conversions as conv


def data_storer(data: tuple, formatter: str = '%1.5e') -> None:
    '''
    Save the given data from the tuples in a csv file. It synchronizes the
    measurement series with each other.

    Parameters
    ----------
    data : tuple
        Data tuple to be saved.
    formatter : str, optional
        In which format the results should be saved. The default is '%1.5e'.

    Returns
    -------
    None
    '''
    if not os.path.exists('output'):
        os.makedirs('output')
        print('The directory "output" has been created automatically.')
        print('In these will be the saved energy files.\n')

    (E_rot, rot_str, E_trans,
     trans_str, E_kin, kin_str, filenames) = output_array(data)
    to_day = str(datetime.now())
    to_day = to_day[:19].replace('-', '_').replace(' ', '-').replace(':', '_')
    print('')
    if 'Gyroscope' in filenames or 'AccGyr' in filenames:
        _path = f'output/E_rot_{to_day}.csv'
        (_, y) = E_rot.shape()
        E_rot[1:, 1] /= y-2
        save_to_file(_path, E_rot, rot_str, formatter)

    if 'Accelerometer' in filenames or 'AccGyr' in filenames:
        _path = f'output/E_trans_{to_day}.csv'
        (_, y) = E_trans.shape()
        E_trans[1:, 1] /= y-2
        save_to_file(_path, E_trans, trans_str, formatter)

    if 'AccGyr' in filenames:
        _path = f'output/E_kin_{to_day}.csv'
        (_, y) = E_kin.shape()
        E_kin[1:, 1] /= y-2
        save_to_file(_path, E_kin, kin_str, formatter)


def save_to_file(_path: str, data_to_save: np.ndarray, header: str,
                 formatter: str = '%1.5e') -> None:
    '''
    Save the given data in a csv file.

    Parameters
    ----------
    _path : str
        Location where the file should be saved.
    data_to_save : np.ndarray
        Data to be saved.
    header : str
        What should be in the header of the file..
    formatter : str, optinal
        In which format the results should be saved. The default is '%1.5e'.

    Returns
    -------
    None
    '''
    try:
        with open(_path, 'x'):
            np.savetxt(_path, data_to_save, delimiter=", ", header=header,
                       fmt=formatter)

    except FileExistsError:
        new_path = _path.replace('.csv', '')
        new_path += f'_{np.random.randint(0, 1000)}.csv'
        print(f'{_path} file exists, a new file name has been created.')
        print(f'New file name: {new_path}.')
        save_to_file(new_path, data_to_save, header, formatter)

    finally:
        print(f'{_path} saved.')


def output_array(data: tuple) -> (np.ndarray, str, np.ndarray, str, np.ndarray,
                                  str, str):
    '''
    Creates the array what should be saved.

    Parameters
    ----------
    data : tuple
        Data tuple to be saved

    Returns
    -------
    E_rot_all : np.ndarray
        Data of all measurements of rotational energy.
    rot_str : str
        Header string for the rotational energy.
    E_trans_all : np.ndarray
        Data of all measurements of the translation energy.
    trans_str : str
        Header string for the translation energy.
    E_kin_all : np.ndarray
        Data of all measurements of kenetic energy.
    kin_str : str
        Header string for the kenetic energy.
    filenames : str
        Name of the files used
    '''
    _list = [0, 0, 0]
    rot_str = 'Time in s, rotational energy in J, '
    trans_str = 'Time in s, translation energy in J, '
    kin_str = 'Time in s, kenetic energy in J, '
    filenames = ''
    E_trans_all = E_rot_all = E_kin_all = np.array([])
    for singel_data in data:
        filename = singel_data[0]
        filenames += filename
        t = singel_data[1]
        E_trans = singel_data[2]
        E_rot = singel_data[3]
        E_kin = singel_data[4]
        if 'Gyroscope' in filename:
            (E_rot_all, rot_str,
             _list[0]) = data_array_test(E_rot_all, E_rot, t, _list[0],
                                         rot_str, 'G', filename)

        elif 'Accelerometer' in filename:
            (E_trans_all, trans_str,
             _list[1]) = data_array_test(E_trans_all, E_trans, t, _list[1],
                                         trans_str, 'A', filename)

        elif 'AccGyr' in filename:
            (E_rot_all, rot_str,
             _list[0]) = data_array_test(E_rot_all, E_rot, t, _list[0],
                                         rot_str, 'AccGyr', filename)
            (E_trans_all, trans_str,
             _list[1]) = data_array_test(E_trans_all, E_trans, t, _list[1],
                                         trans_str, 'AccGyr', filename)
            (E_kin_all, kin_str,
             _list[2]) = data_array_test(E_kin_all, E_kin, t, _list[2],
                                         kin_str, 'AccGyr', filename)
        else:
            print(f'No saving method is known for {filename}.')

    rot_str = str_edit(rot_str)
    trans_str = str_edit(trans_str)
    kin_str = str_edit(kin_str)
    return (E_rot_all, rot_str, E_trans_all, trans_str, E_kin_all, kin_str,
            filenames)


def data_array_test(E_all: np.ndarray, E_add: np.ndarray, t: np.ndarray,
                    count: int, name_str: str, check: str,
                    filename: str) -> (np.ndarray, str, int):
    '''
    Tests if an array already exists for this energy.

    Parameters
    ----------
    E_all : np.ndarray
        The energy array to which E_all is to be added.
    E_add : np.ndarray
        The energy array to be added.
    t : np.ndarray
        Time when E_âdd has been measured.
    count : int
        Whether this array exists.
    name_str : str
        Header string for the used array.
    check : str
        Which energy is processed.
    filename : str
        Name of the file used.

    Returns
    -------
    E_all : np.ndarray
        Data of the energies processed so far.
    name_str : str
        Header string for the used array.
    check : str
        Which energy is processed.
    '''
    if count == 0:
        count = 1
        (E_all,
         name_str) = data_array_first(t, E_add, filename, check, name_str)

    else:
        (E_all,
         name_str) = data_array_add(E_all, t, E_add, filename, check,
                                    name_str)

    return (E_all, name_str, count)


def data_array_first(t: np.ndarray, E_add: np.ndarray, filename: str,
                     check: str, name_str: str) -> (np.ndarray, str):
    '''
    Tests if an array already exists for this energy.

    Parameters
    ----------
    t : np.ndarray
        Time when E_âdd has been measured.
    E_add : np.ndarray
        The energy array to be added.
    filename : str
        Name of the file used.
    check : str
        Which energy is processed.
    name_str : str
        Header string for the used array.

    Returns
    -------
    E_all : np.ndarray
        Data of the energies processed so far.
    name_str : str
        Header string for the used array.
    '''
    E_all = t
    name_str = conv.string(name_str, filename, check)
    E_all = np.c_[E_all, E_add]  # for the total energy
    return (np.c_[E_all, E_add], name_str)


def data_array_add(E_all: np.ndarray, t: np.ndarray, E_add: np.ndarray,
                   filename: str, check: str,
                   name_str: str) -> (np.ndarray, str):
    '''
    Tests if an array already exists for this energy.

    Parameters
    ----------
    E_all : np.ndarray
        The energy array to which E_all is to be added.
    t : np.ndarray
        Time when E_âdd has been measured.
    E_add : np.ndarray
        The energy array to be added.
    filename : str
        Name of the file used.
    check : str
        Which energy is processed.
    name_str : str
        Header string for the used array.

    Returns
    -------
    E_all : np.ndarray
        Data of the energies processed so far.
    name_str : str
        Header string for the used array.
    '''
    E_now = np.interp(E_all[:, 0], t, E_add)
    E_all[:, 1] += E_now
    name_str = conv.string(name_str, filename, check)
    return (np.c_[E_all, E_now], name_str)


def str_edit(name_str: str) -> str:
    '''
    Converts ',' to ', '.
    '''
    name_str = name_str.replace(',', ', ')
    return name_str[:-2]
