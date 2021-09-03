# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:22:43 2021
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

from dataclasses import dataclass
import numpy as np

@dataclass()
class Q:
    w: float = 0
    i: float = 0
    j: float = 0
    k: float = 0

    def __init__(self, w, i, j, k):
        self.w = w
        self.i = i
        self.j = j
        self.k = k

    def normalized(self):
        '''
        Normalizes the quaternion.
        '''
        n = self.norm()
        return Q(self.w/n, self.i/n, self.j/n, self.k/n)

    def conjugation(self):
        '''
        Conjugation the quaterion
        '''
        return Q(self.w, -self.i, -self.j, -self.k)

    def scalarproduct(self, other):
        '''
        Scala product of two quaterions
        '''
        q = self
        q.__add__(other)
        return (q.w+q.i+q.j+q.k)

    def invers(self):
        '''
        Invers the quaterion
        '''
        self.normalized()
        return self.conjugation()

    def euler_angels(self):
        '''
        Calculates the Euler angles for the quaternion.
        '''
        q = self.normalized()
        a = q.w
        b = q.i
        c = q.j
        d = q.k
        alpha = np.arctan2(2.0*(b*c+a*d), (a**2+b**2-c**2-d**2)) * 180.0/np.pi
        beta = np.arcsin(2.0*(a*c-b*d)) * 180.0/np.pi
        gamma = -np.arctan2(2.0*(c*d+a*b), -(a**2-b**2-c**2+d**2)) * 180.0/np.pi
        return np.array([alpha, beta, gamma])

    def __eq__(self, other):
        '''
        Tests if the norm of the two quations is the same.
        '''
        if type(other) != type(self):
            raise TypeError(f'unsupported operand type(s) for == : {type(self)} and {type(other)}')
        if self.norm() == other.norm():
            return True
        else:
            return False

    def __ne__(self, other):
        '''
        Tests if the norm of the two quations is not the same.
        '''
        if type(other) != type(self):
            raise TypeError(f'unsupported operand type(s) for != : {type(self)} and {type(other)}')

        if self.norm() != other.norm():
            return True
        else:
            return False

    def __lt__(self, other):
        '''
        Tests if the norm of the first quation is smaller than the norm of the second quation.
        '''
        if type(other) != type(self):
            raise TypeError(f'unsupported operand type(s) for < : {type(self)} and {type(other)}')

        if self.norm() < other.norm():
            return True
        else:
            return False

    def __gt__(self, other):
        '''
        Tests if the norm of the first quation is greater than the norm of the second quation.
        '''
        if type(other) != type(self):
            raise TypeError(f'unsupported operand type(s) for > : {type(self)} and {type(other)}')

        if self.norm() > other.norm():
            return True
        else:
            return False

    def __le__(self, other):
        '''
        Tests if the norm of the first quation is less than or equal to the norm of the second quation.
        '''
        if type(other) != type(self):
            raise TypeError(f'unsupported operand type(s) for <= : {type(self)} and {type(other)}')

        if self.norm() <= other.norm():
            return True
        else:
            return False

    def __ge__(self, other):
        '''
        Tests if the norm of the first quation is greater than or equal to the norm of the second quation.
        '''
        if type(other) != type(self):
            raise TypeError(f'unsupported operand type(s) for >=: {type(self)} and {type(other)}')

        if self.norm() >= other.norm():
            return True
        else:
            return False

    def __str__(self):
        if self.w>=0 and self.i>=0 and self.j>=0 and self.k>=0:
            return f'{self.w}+{self.i}i+{self.j}j+{self.k}k'
        elif self.w>=0 and self.i>=0 and self.j>=0 and self.k<0:
            return f'{self.w}+{self.i}i+{self.j}j{self.k}k'
        elif self.w>=0 and self.i>=0 and self.j<0 and self.k>=0:
            return f'{self.w}+{self.i}i{self.j}+j{self.k}k'
        elif self.w>=0 and self.i>=0 and self.j<0 and self.k<0:
            return f'{self.w}+{self.i}i{self.j}j{self.k}k'
        elif self.w>=0 and self.i<0 and self.j>=0 and self.k>=0:
            return f'{self.w}{self.i}i+{self.j}j+{self.k}k'
        elif self.w>=0 and self.i<0 and self.j>=0 and self.k<0:
            return f'{self.w}{self.i}i+{self.j}j{self.k}k'
        elif self.w>=0 and self.i<0 and self.j<0 and self.k>=0:
            return f'{self.w}{self.i}i{self.j}+j{self.k}k'
        elif self.w>=0 and self.i<0 and self.j<0 and self.k<0:
            return f'{self.w}{self.i}i{self.j}j{self.k}k'
        elif self.w<0 and self.i>=0 and self.j>=0 and self.k>=0:
            return f'{self.w}+{self.i}i+{self.j}+j{self.k}k'
        elif self.w<0 and self.i>=0 and self.j>=0 and self.k<0:
            return f'{self.w}+{self.i}i+{self.j}j{self.k}k'
        elif self.w<0 and self.i>=0 and self.j<0 and self.k>=0:
            return f'{self.w}+{self.i}i{self.j}+j{self.k}k'
        elif self.w<0 and self.i>=0 and self.j<0 and self.k<0:
            return f'{self.w}+{self.i}i{self.j}j{self.k}k'
        elif self.w<0 and self.i<0 and self.j>=0 and self.k>=0:
            return f'{self.w}{self.i}i+{self.j}j+{self.k}k'
        elif self.w<0 and self.i<0 and self.j>=0 and self.k<0:
            return f'{self.w}{self.i}i+{self.j}j{self.k}k'
        elif self.w<0 and self.i<0 and self.j<0 and self.k>=0:
            return f'{self.w}{self.i}i{self.j}+j{self.k}k'
        elif self.w<0 and self.i<0 and self.j<0 and self.k<0:
            return f'{self.w}{self.i}i{self.j}j{self.k}k'

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        if self != 0:
            return True
        else:
            return False

    def __add__(self, other):
        '''
        Adds two quaternions.
        '''
        if type(other) == type(self):
            return Q(self.w+other.w, self.i+other.i, self.j+other.j, self.k+other.k)
        elif str(type(other)) in __listtyp__():
            return Q(self.w+other, self.i+other, self.j+other, self.k+other)
        else:
            raise TypeError(f'unsupported operand type(s) for +: {type(self)} and {type(other)}')

    def __sub__(self, other):
        '''
        Subtracts two quations
        '''
        if type(other) == type(self):
            return Q(self.w-other.w, self.i-other.i, self.j-other.j, self.k-other.k)
        elif str(type(other)) in __listtyp__():
            return Q(self.w-other, self.i-other, self.j-other, self.k-other)
        else:
            raise TypeError(f'unsupported operand type(s) for -: {type(self)} and {type(other)}')

    def __mul__(self, other):
        '''
        Multiplies two quaternions.
        '''
        if type(other) == type(self):
            return Q(self.w*other.w-self.i*other.i-self.j*other.j-self.k*other.k,
                 self.w*other.i+self.i*other.w+self.j*other.k-self.k*other.j,
                 self.w*other.j-self.i*other.k+self.j*other.w+self.k*other.i,
                 self.w*other.k+self.i*other.j-self.j*other.i+self.k*other.w)
        elif str(type(other)) in __listtyp__():
            return Q(self.w*other,self.i*other,self.j*other,self.k*other)
        else:
            raise TypeError(f'unsupported operand type(s) for *: {type(self)} and {type(other)}')

    def __pow__(self, other):
        if str(type(other)) in __listtyp__():
            return Q(self.w**other,self.i**other,self.j**other,self.k**other)
        else:
            raise TypeError(f'unsupported operand type(s) for **: {type(self)} and {type(other)}')


    def __truediv__(self, other):
        '''
        Attention both quations are normalized and the second quation is inverted!
        '''
        if type(other) == type(self):
            self.normalized()
            return self.__mul__(other.invers())
        elif str(type(other)) in __listtyp__():
            return Q(self.w/other,self.i/other,self.j/other,self.k/other)
        else:
            raise TypeError(f'unsupported operand type(s) for /: {type(self)} and {type(other)}')

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __ipow__(self, other):
        return self.__pow__(other)

    def __pos__(self):
        return self

    def __neg__(self):
        return Q(-self.w, -self.i, -self.j, -self.k)

    def __abs__(self):
        return np.sqrt(self.w**2+self.i**2+self.j**2+self.k**2)

    def __invert(self):
        return Q(~self.w,~self.i,~self.j,~self.k)

    def __round__(self, ndigits):
        return Q(round(self.w,ndigits),round(self.i,ndigits),
                 round(self.j,ndigits),round(self.k,ndigits))

def __listtyp__():
    return [str(int), str(float),
            '''<class 'numpy.int8'>''','''<class 'numpy.int16'>''',
            '''<class 'numpy.int32'>''','''<class 'numpy.int64'>''',
            '''<class 'numpy.uint8'>''','''<class 'numpy.uint16'>''',
            '''<class 'numpy.uint32'>''','''<class 'numpy.uint64'>''',
            '''<class 'numpy.float16'>''','''<class 'numpy.float32'>''',
            '''<class 'numpy.float64'>''','''<class 'numpy.float128'>''']

