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
import statistics as stats


# Prints a histogram based on the values "values".
#
# values:         the values to be displayed in a histogram
# bucket_count:   the number of buckets to be used. It is also the number of bars in the histogram
# height:         the height of the histogram in lines
# width_factor:   the width of the bars in characters
# spacing_factor: the spacing between the bars in characters
def display_histogram(values, bucket_number, height, width_factor, spacing_factor):
    histogram = values_to_buckets(values, bucket_number)
    largest = stats.get_largest_number(histogram)
    metric = largest / height
    spacing = " " * spacing_factor

    for i in range(0, height):
        index = height - i
        value = index * metric

        for n in histogram:
            if n >= value:
                print("|" * width_factor + spacing, end="")
            else:
                print(" " * width_factor + spacing, end="")
        print("")


# Creates a list of buckets, their filling is based on the values "values".
#
# values:       the values used to fill the buckets
# bucket_count: the number of buckets to be used. It is also the length of the returned array.
# returns:      a list of buckets, each bucket is an integer representing the number of values in the bucket
def values_to_buckets(values, bucket_count):
    buckets = [0 for i in range(0, bucket_count)]
    smallest = stats.get_smallest_number(values)
    largest = stats.get_largest_number(values)
    value_per_bucket = np.absolute(largest - smallest) / bucket_count

    for value in values:
        bucket_index = int((value - smallest) / value_per_bucket)
        if bucket_index == bucket_count:
            bucket_index -= 1
        buckets[bucket_index] += 1

    return buckets
