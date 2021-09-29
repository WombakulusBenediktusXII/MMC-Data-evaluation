# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:36:05 2021
@author: Anton
Version: v0.3-alpha

In the config_parser module, the config.ini is read in and processed..
These are:
    get_config, _main_config, _acc_config, _gyr_config, _graph_config,
    _int_config, float_config, _array_config, _list_config, _bool_config,
    _str_config, _test_list
"""

import numpy as np
from configparser import ConfigParser


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
        main_dict = _main_config(config['MAIN'])
    except KeyError:
        main_dict = _main_config(list())

    try:
        acc_dict = _acc_config(config['ACCELEROMETER'])
    except KeyError:
        acc_dict = _acc_config(list())

    try:
        gyr_dict = _gyr_config(config['GYROSCOPE'])
    except KeyError:
        gyr_dict = _gyr_config(list())

    try:
        graph_dict = _graph_config(config['GRAPH'])
    except KeyError:
        graph_dict = _graph_config(list())

    acc_dict.update({'r': main_dict['r']})
    acc_dict.update({'m': main_dict['m']})
    gyr_dict.update({'r': main_dict['r']})
    gyr_dict.update({'m': main_dict['m']})
    return (main_dict, acc_dict, gyr_dict, graph_dict)


def _main_config(config: ConfigParser) -> dict:
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
    _float_config(main_dict, 'm', 0.01496)
    _float_config(main_dict, 'r', 0.02925)
    _list_config(main_dict, 'names', [''])
    _list_config(main_dict, 'filenames', ['input/Test_Datei.csv'])
    _list_config(main_dict, 'measurements', [''])
    _bool_config(main_dict, 'filenames_auto', True)
    _bool_config(main_dict, 'do_output', False)
    return main_dict


def _acc_config(config: ConfigParser) -> dict:
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
    _int_config(acc_dict, 'degree_of_spline', 50)
    _float_config(acc_dict, 'error', 0.001)
    _float_config(acc_dict, 'smoothes', 0.8)
    _array_config(acc_dict, 'sensorpos',
                 np.array([1.2, 7.4, 4.5]))
    _array_config(acc_dict, 'start_velocity', np.array([0, 0, 0]))
    _bool_config(acc_dict, 'in_g', True)
    _bool_config(acc_dict, 'g_interfered', True)
    _bool_config(acc_dict, 'trajectory', False)
    _str_config(acc_dict, 'integration_mode', 'a')
    return acc_dict


def _gyr_config(config: ConfigParser) -> dict:
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
    _int_config(gyr_dict, 'degree_of_spline', 5)
    _float_config(gyr_dict, 'error', 0.01)
    _float_config(gyr_dict, 'smoothes', 0.8)
    _array_config(gyr_dict, 'start_rotation', np.array([0, 0, 0]))
    _bool_config(gyr_dict, 'in_grad', True)
    _str_config(gyr_dict, 'integration_mode', 's')
    return gyr_dict


def _graph_config(config: ConfigParser) -> dict:
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
    if 'formatter' in graph_dict:
        graph_dict['formatter'] = '%' + graph_dict['formatter']
    else:
        graph_dict.update({'formatter': '%1.2e'})
    return graph_dict


def _int_config(test_dict: dict, config: str, default: int) -> None:
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
    if config in test_dict:
        test_dict[config] = int(test_dict[config])
    else:
        test_dict.update({config: default})
    pass


def _float_config(test_dict: dict, config: str, default: float) -> None:
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
    if config in test_dict:
        test_dict[config] = float(test_dict[config])
    else:
        test_dict.update({config: default})
    pass


def _array_config(test_dict: dict, config: str, default: np.ndarray) -> None:
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
    if config in test_dict:
        test_dict[config] = np.array(test_dict[config].replace(' ', '').split(','), dtype=float)
    else:
        test_dict.update({config: default})
    pass


def _list_config(test_dict: dict, config: str, default: list) -> None:
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
    if config in test_dict:
        test_str = test_dict[config].replace(' ', '')
        test_dict[config] = test_str.split(',')
    else:
        test_dict.update({config: default})
    pass


def _bool_config(test_dict: dict, config: str, default: bool) -> None:
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
    if config in test_dict:
        if test_dict[config] in _test_list(True):
            test_dict[config] = True
        elif test_dict[config] in _test_list(False):
            test_dict[config] = False
        else:
            _test_list(test_dict[config], config)
    else:
        test_dict.update({config: default})
    pass


def _str_config(test_dict: bool, config: str, default: str) -> None:
    '''
    Tests if the key is present in the dictionary, if not the default value is used.

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
    if config in test_dict:
        pass
    else:
        test_dict.update({config: default})
    pass


def _test_list(test: bool, config: str = '') -> list:
    '''
    Gives a list of values to be interpreted as true or false.

    Parameters
    ----------
    test : bool
        Whether to return the true or false list..
    config : str, optional
        Which key is tested. Is there to generate a better error message if the
        input cannot be interpreted as true or false.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    if test is True:
        return ['True', 'Yes', '1', 'Y', 'true', 'yes', 'y']
    elif test is False:
        return ['False', 'No', '0', 'F', 'false', 'no', 'f']
    else:
        conf = config
        raise TypeError(f'The given value for {conf} is neither true or false')
