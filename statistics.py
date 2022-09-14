"""
    Uncertainty-Analysis-Toolkit-for-Metos3d can be used to generate option files for Metos3d
    and interpret Metos3d output data.
    Copyright (C) 2022  Tom L. Hauschild

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np


# Estimates important attributes of the "data" interpreted as being lognormal distributed.
#
# data:    the data (array of numbers) to be analyzed
# returns: mu, sigma, estimated value, variance
def estimate_lognorm_data_values(data):
    # estimating mu
    mu = np.sum(np.log(data)) / len(data)
    # estimating sigma^2
    s2 = np.sum((np.log(data) - mu) ** 2) / len(data)
    # calculating expected value
    e = np.exp(mu + 0.5 * s2)
    # calculating variance
    v = np.exp(2 * mu + s2) * (np.exp(s2) - 1)
    return mu, np.sqrt(s2), e, v


# Calculates the estimated value and the variance of a lognormal distribution with the given parameters mu and s.
#
# mu:      the mu
# s:       the sigma
# returns: the estimated value and the variance
def lognorm_values(mu, s):
    s2 = s ** 2
    e = np.exp(mu + 0.5 * s2)
    v = np.exp(2 * mu + s2) * (np.exp(s2) - 1)
    return e, v


# Calculates the largest number in the array "array".
#
# array:   the array to be analyzed
# returns: the largest number in the array
def get_largest_number(array):
    largest_number = array[0]
    for element in array:
        if element > largest_number:
            largest_number = element
    return largest_number


# Calculates the smallest number in the array "array".
#
# array:   the array to be analyzed
# returns: the smallest number in the array
def get_smallest_number(array):
    smallest_number = array[0]
    for element in array:
        if element < smallest_number:
            smallest_number = element
    return smallest_number


# Normalizes the array "values" to the range [0, 1].
#
# values:  the array to be normalized
# returns: the normalized array
def normalize(values):
    max = get_largest_number(values)
    min = get_smallest_number(values)
    new_values = [0 for i in range(len(values))]
    for i in range(0, len(values)):
        new_values[i] = (values[i] - min) / (max - min)
    return new_values


# Calculates the empirical correlation coefficient of the two arrays "values1" and "values2".
#
# values1: the first array
# values2: the second array
# returns: the empirical correlation coefficient
def empirical_correlation_coefficient(values1, values2):
    n = len(values1)
    r0 = np.sum(np.multiply(values1, values2)) - ((1/n) * np.sum(values1) * np.sum(values2))
    r10 = np.sum(np.multiply(values1, values1)) - ((1/n) * np.sum(values1) ** 2)
    r11 = np.sum(np.multiply(values2, values2)) - ((1/n) * np.sum(values2) ** 2)
    r1 = np.sqrt(r10 * r11)
    return r0 / r1
