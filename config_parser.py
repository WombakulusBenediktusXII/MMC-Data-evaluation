# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 2021
@author: Anton
Version: v0.4-beta

In the config_parser module, the config.ini is read in and processed.
These are:
    get_config, _main_config, _acc_config, _gyr_config, _graph_config,
    int_config, float_config, array_config, list_config, bool_config,
    str_config, test_list
"""

from configparser import ConfigParser
from os import cpu_count
import numpy as np


def get_config(filename: str = 'config.ini') -> dict:
    '''
    Reads the config file and outputs the data as a dictionary. If a section
    key is not given, the dictionaries are created with the default values.

    Parameters
    ----------
    filename : str, optional
        Name of the config file. The default is 'config.ini'.

    Returns
    -------
    main_dict : dict
        Dictionary which has the constants for main.
    acc_dict : dict
        Dictionary which has the constants for the acelerometer.
    gyr_dict : dict
        Dictionary which has the constants for the gyroscope.
    graph_dict : dict
        Dictionary which has the constants for the diagrams.
    '''
    config = ConfigParser()
    config.read(filename)

    try:
        main_dict = main_config(config['MAIN'])
    except KeyError:
        main_dict = main_config(list())

    try:
        acc_dict = acc_config(config['ACCELEROMETER'])
    except KeyError:
        acc_dict = acc_config(list())

    try:
        gyr_dict = gyr_config(config['GYROSCOPE'])
    except KeyError:
        gyr_dict = gyr_config(list())

    try:
        graph_dict = graph_config(config['GRAPH'])
    except KeyError:
        graph_dict = graph_config(list())

    acc_dict.update({'r': main_dict['r']})
    acc_dict.update({'m': main_dict['m']})
    gyr_dict.update({'r': main_dict['r']})
    gyr_dict.update({'m': main_dict['m']})
    return (main_dict, acc_dict, gyr_dict, graph_dict)


def main_config(config: ConfigParser) -> dict:
    '''
    Produce the main dictionary.

    Parameters
    ----------
    config : ConfigParser
        Main section of the config file.

    Returns
    -------
    main_dict : dict
        Main dictionaray with read in config data.

    '''
    main_dict = dict(config)
    int_config(main_dict, 'max_processes', cpu_count())
    float_config(main_dict, 'm', 0.023)
    float_config(main_dict, 'r', 0.019)
    list_config(main_dict, 'names', [''])
    list_config(main_dict, 'filenames', ['input/Test_Datei.csv'])
    list_config(main_dict, 'measurements', [''])
    bool_config(main_dict, 'filenames_auto', True)
    bool_config(main_dict, 'save_output', False)
    bool_config(main_dict, 'multi_processing', 'AUTO')
    formatter_config(main_dict, 'save_formatter', '%1.5e')
    return main_dict


def acc_config(config: ConfigParser) -> dict:
    '''
    Produce the accelerometer dictionary.

    Parameters
    ----------
    config : ConfigParser
        Main section of the config file.

    Returns
    -------
    acc_dict : dict
        Accelerometer dictionaray with read in config data.

    '''
    acc_dict = dict(config)
    int_config(acc_dict, 'degree_of_spline', 50)
    float_config(acc_dict, 'error', 0.001)
    float_config(acc_dict, 'smoothes', 0.8)
    array_config(acc_dict, 'sensorpos',
                 np.array([0.0012, 0.0074, 0.0045]))
    array_config(acc_dict, 'start_velocity', np.array([0, 0, 0]))
    bool_config(acc_dict, 'in_g', True)
    bool_config(acc_dict, 'g_interfered', True)
    bool_config(acc_dict, 'trajectory', False)
    str_config(acc_dict, 'integration_mode', 'a')
    return acc_dict


def gyr_config(config: ConfigParser) -> dict:
    '''
    Produce the gyroscope dictionary.

    Parameters
    ----------
    config : ConfigParser
        Main section of the config file.

    Returns
    -------
    gyr_dict : dict
        Accelerometer dictionaray with read in config data.

    '''
    gyr_dict = dict(config)
    int_config(gyr_dict, 'degree_of_spline', 5)
    float_config(gyr_dict, 'error', 0.01)
    float_config(gyr_dict, 'smoothes', 0.8)
    array_config(gyr_dict, 'start_rotation', np.array([0, 0, 0]))
    bool_config(gyr_dict, 'in_grad', True)
    str_config(gyr_dict, 'integration_mode', 's')
    return gyr_dict


def graph_config(config: ConfigParser) -> dict:
    '''
    Produce the graph dictionary.

    Parameters
    ----------
    config : ConfigParser
        Main section of the config file.

    Returns
    -------
    graph_dict : dict
        Graph dictionaray with read in config data.

    '''
    graph_dict = dict(config)
    bool_config(graph_dict, 'do_graph', True)
    bool_config(graph_dict, 'save_graph', False)
    formatter_config(graph_dict, 'formatter', '%1.2e')
    return graph_dict


def int_config(test_dict: dict, config: str, default: int) -> None:
    '''
    Converts the read string into an integer and updates the dictionary.

    Parameters
    ----------
    test_dict : dict
        The dictionary to be changed.
    config : str
        Which key to retrieve.
    default : int
        The default value for this key if it does not exist.

    Returns
    -------
    None
    '''
    try:
        test_dict[config] = int(test_dict[config])
    except KeyError:
        test_dict.update({config: default})
        print(f'The default value has been selected for {config}. This is {default}.')


def float_config(test_dict: dict, config: str, default: float) -> None:
    '''
    Converts the read string into an float and updates the dictionary.

    Parameters
    ----------
    test_dict : dict
        The dictionary to be changed.
    config : str
        Which key to retrieve.
    default : float
        The default value for this key if it does not exist.

    Returns
    -------
    None
    '''
    try:
        test_dict[config] = float(test_dict[config])
    except KeyError:
        test_dict.update({config: default})
        print(f'The default value has been selected for {config}. This is {default}.')


def array_config(test_dict: dict, config: str, default: np.ndarray) -> None:
    '''
    Converts the read string into an np.ndarray and updates the dictionary.

    Parameters
    ----------
    test_dict : dict
        The dictionary to be changed.
    config : str
        Which key to retrieve.
    default : np.ndarray
        The default value for this key if it does not exist.

    Returns
    -------
    None
    '''
    try:
        test_dict[config] = np.array(test_dict[config].replace(' ', '').split(','),
                                     dtype=float)
    except KeyError:
        test_dict.update({config: default})
        print(f'The default value has been selected for {config}. This is {default}.')


def list_config(test_dict: dict, config: str, default: list) -> None:
    '''
    Converts the read string into an list and updates the dictionary.

    Parameters
    ----------
    test_dict : dict
        The dictionary to be changed.
    config : str
        Which key to retrieve.
    default : list
        The default value for this key if it does not exist.

    Returns
    -------
    None
    '''
    try:
        test_str = test_dict[config].replace(' ', '')
        test_dict[config] = test_str.split(',')
    except KeyError:
        test_dict.update({config: default})
        print(f'The default value has been selected for {config}. This is {default}.')


def bool_config(test_dict: dict, config: str, default: bool) -> None:
    '''
    Converts the read string into an boolean and updates the dictionary.

    Parameters
    ----------
    test_dict : dict
        The dictionary to be changed.
    config : str
        Which key to retrieve.
    default : bool
        The default value for this key if it does not exist.

    Returns
    -------
    None
    '''
    try:
        if test_dict[config] in test_list(True):
            test_dict[config] = True
        elif test_dict[config] in test_list(False):
            test_dict[config] = False
        elif test_dict[config] in ['Auto', 'AUTO', 'auto']:
            test_dict[config] = None
        else:
            test_list(test_dict[config], config)
    except KeyError:
        test_dict.update({config: default})
        print(f'The default value has been selected for {config}. This is {default}.')


def str_config(test_dict: dict, config: str, default: str) -> None:
    '''
    Tests if the key is present in the dictionary, if not the default value is
    used.

    Parameters
    ----------
    test_dict : dict
        The dictionary to be changed.
    config : str
        Which key to retrieve.
    default : str
        The default value for this key if it does not exist.

    Returns
    -------
    None
    '''
    if config not in test_dict:
        test_dict.update({config: default})
        print(f'The default value has been selected for {config}. This is {default}.')


def formatter_config(test_dict: dict, config: str, default: str) -> None:
    '''
    Converts the read string into an formatter and updates the dictionary.

    Parameters
    ----------
    test_dict : dict
        The dictionary to be changed.
    config : str
        Which key to retrieve.
    default : str
        The default value for this key if it does not exist.

    Returns
    -------
    None
    '''
    try:
        test_dict[config] = '%' + test_dict[config]
    except KeyError:
        test_dict.update({config: default})
        print(f'The default value has been selected for {config}. This is {default}.')


def test_list(test: bool, config: str = '') -> list:
    '''
    Gives a list of values to be interpreted as true or false.

    Parameters
    ----------
    test : bool
        Whether to return the true or false list..
    config : str, optional
        Which key is tested. Is there to generate a better error message if the
        input cannot be interpreted as true or false.

    Raise
    -----
    TypeError
        If the given config key cannot be interpreted as True or False.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    if test is True:
        res = ['True', 'Yes', '1', 'Y', 'true', 'yes', 'y']
    elif test is False:
        res = ['False', 'No', '0', 'F', 'false', 'no', 'f']
    else:
        raise TypeError(f'The given value for {config} is neither true or false')

    return res


if __name__ == '__main__':
    for n in get_config():
        print(n)
