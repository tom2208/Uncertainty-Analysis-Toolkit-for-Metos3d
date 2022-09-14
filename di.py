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

import csv

import petsc_mod as pe
import mpg
import statistics as stats
from scipy.stats import ks_2samp
from scipy.stats import anderson
import numpy as np
import matplotlib.pyplot as plt


# Reads a csv file and returns its content as a list of lists.
#
# path: the path of the csv file
def values_from_csv(path):
    results = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            results.append(row)
    return results


# Reads multiple .petsc files, calculates the sum of the "layer"-th layer for each file and returns this sums as a list.
#
# file_name: the path to the files containing %i% as an placeholder for the index of the file
#            (index in range of 0 to n)
# layer:     the layer to be summed up
# n:         the number of files to be read
# return:    a list containing the sums of the "layer"-th layer for each file
def generate_value_array(file_name, layer, n):
    values = [0 for i in range(n)]

    for i in range(n):
        values[i] = get_value_from_file(file_name.replace("%i%", str(i)), layer)

    return values


# Calculates the sum of the "layer"-th layer for a given file and returns it.
#
# file_name: the path to the file
# layer:     the layer to be summed up
# return:    the sum of the "layer"-th layer for the given file
def get_value_from_file(file, layer):
    lsm = pe.read_PETSc_matrix('landSeaMask.petsc')
    v = pe.read_PETSc_vec(file)
    v3d, n1, n2, n3 = pe.reshape_vector_to_3d(lsm, v)
    if is_rectangle:
        x1 = rectangle[0]
        x2 = rectangle[1]
        y1 = rectangle[2]
        y2 = rectangle[3]

    layer_sum = 0
    for x in range(len(v3d[:, :, layer])):
        for y in range(len(v3d[:, :, layer][x])):
            if v3d[x][y][layer] == v3d[x][y][layer]:
                if (not is_rectangle) or (x1 <= x <= x2 and y1 <= y <= y2):
                    layer_sum += v3d[x][y][layer]

    return layer_sum


# Prints out an analysis of the given data. Including Kolmogorov-Smirnov and Anderson-Darling test results and the
# attributes of the data interpreted as a lognormal and distribution.
#
# values: the data to be analyzed.
# mu:     the mu of the data interpreted as a lognormal distribution
# s:      the sigma of the data interpreted as a lognormal distribution
# e:      the expected value of the data interpreted as a lognormal distribution
# v:      the variance of the data interpreted as a lognormal distribution
def print_attributes(values, mu, s, e, v):
    mpg.print_double_seperator()
    mpg.print_info("Analytics of the values:")
    mpg.print_seperator()

    data_lognorm = np.random.lognormal(mu, s, 1000000)
    data_normal = np.random.normal(np.mean(values), np.var(values), 1000000)

    # K-S-Tests
    result = ks_2samp(values, data_lognorm)
    result2 = ks_2samp(values, data_normal)
    mpg.print_info("Kolmogorov-Smirnov test results for a lognormal distribution:")
    mpg.print_info("statistic: \t" + str(result[0]))
    mpg.print_info("p-value: \t" + str(result[1]))

    mpg.print_seperator()

    mpg.print_info("Kolmogorov-Smirnov test results for a normal distribution:")
    mpg.print_info("statistic: \t" + str(result2[0]))
    mpg.print_info("p-value: \t" + str(result2[1]))

    mpg.print_seperator()

    # A-D-Test
    a_d_result = anderson(values)
    mpg.print_info("Anderon-Darling test results:")
    mpg.print_info("statistic:\t\t" + str(a_d_result[0]))
    mpg.print_info("critical values:\t\t" + str(a_d_result[1]))
    mpg.print_info("significance level:\t" + str(a_d_result[2]))

    mpg.print_seperator()
    mpg.print_info("values for a lognormal distribution:")
    mpg.print_info("estimated mu:\t\t\t" + str(mu))
    mpg.print_info("estimated sigma:\t\t\t" + str(s))
    mpg.print_info("estimated expected value:\t" + str(e))
    mpg.print_info("estimated variance:\t\t" + str(v))
    mpg.print_seperator()
    mpg.print_info("values for a normal distribution:")
    mpg.print_info("estimated expected value:\t" + str(np.mean(values)))
    mpg.print_info("estimated variance:\t\t" + str(np.var(values)))
    mpg.print_seperator()


#  Plots the given data ("values") as a histogram.
#
# values:       the data to be plotted
# path:         the path to the file where the plot should be saved
# bins:         the number of bins to be used for the histogram
# title:        the title of the plot
# color:        the color of the bars in the histogram
# rotation:     the rotation of the x-axis labels
# plot:         a boolean value indicating if the approximation of the density function should be plotted
# second_color: the color of the approximated density function
# x_axis:       the label of the x-axis
# x_axis2:      the label of the second x-axis
# y_axis:       the label of the y-axis
def generate_histogram(values, path, bins, title, color, rotation, plot, second_color, x_axis, y_axis2, y_axis):
    fig, ax1 = plt.subplots()

    ax1.tick_params(axis='x', rotation=rotation)
    ax1.set_ylabel(y_axis)
    ax1.set_xlabel(x_axis)

    x = np.linspace(stats.get_smallest_number(values), stats.get_largest_number(values), 2000)
    ax1.hist(values, bins=bins, color=color)

    if plot:
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylabel(y_axis2)

        mu, s, e, v = stats.estimate_lognorm_data_values(values)
        y = density_func_lognorm(x, s, mu)
        ax2.plot(x, y, '--', color=second_color)
    plt.title(title)
    plt.tight_layout()
    plt.xticks(rotation=rotation)
    plt.savefig(path)
    mpg.print_success("Histogram saved to " + path)


# Plots the two given data arrays ("values1" and "values2") as a scatter plot and prints out the regression function as
# well as the empirical correlation coefficient.
#
# values1:      the first data array to be plotted
# values2:      the second data array to be plotted
# path:         the path to the file where the plot should be saved
# title:        the title of the plot
# x_axis:       the label of the x-axis
# y_axis:       the label of the y-axis
# regression:   a boolean value indicating if a regression line should be plotted
# color:        the color of the data points
# rotation:     the rotation of the x-axis labels
# second_color: the color of the regression line
def generate_scatter_plot(values1, values2, path, title, x_axis, y_axis, regression, color, rotation, regression_color):
    if len(values1) != len(values2):
        mpg.print_error("The length of the two arrays is not equal! (" + str(len(values1)) + " != " + str(len(values2))
                        + ")")
        mpg.print_error("Hint: the parameter n limits the size of the .petsc arrays")
        exit(0)

    fig = plt.figure()
    f = fig.add_subplot(111)
    f.set_xlabel(x_axis)
    f.set_ylabel(y_axis)
    f.scatter(values1, values2, color=color)
    f.set_title(title)
    b, a = np.polyfit(values1, values2, 1)
    if regression:
        f.plot(values1, np.multiply(b, values1) + a, regression_color)

    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(path)

    mpg.print_double_seperator()
    mpg.print_success("Scatter plot saved to " + path)

    mpg.print_seperator()
    mpg.print_info("linear regression function:")
    mpg.print_info("y = " + str(b) + "x + " + str(a))
    mpg.print_seperator()
    mpg.print_info(
        "empirical correlation coefficient: " + str(stats.empirical_correlation_coefficient(values1, values2)))
    mpg.print_double_seperator()


# Plots a lognormal density function with the given parameters.
#
# mu:    the mu parameter of the lognormal distribution
# sigma: the sigma parameter of the lognormal distribution
# path:  the path to the file where the plot should be saved
# title: the title of the plot
# x_axis: the label of the x-axis
# y_axis: the label of the y-axis
# color: the color of the density function
# rotation: the rotation of the x-axis labels
def plot_lognorm(m, s, b, e, n, path, title, x_axis, y_axis, color, rotation):
    x = np.linspace(b, e, n)
    y = density_func_lognorm(x, s, m)

    plt.plot(x, y, color)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(path)
    mpg.print_success("Plot saved to " + path)


# The density function of a lognormal distribution.
#
# x:      the argument of the density function
# s:      the sigma of the lognormal distribution
# mu:     the mu of the lognormal distribution
# return: the value of the density function at x
def density_func_lognorm(x, s, m):
    return np.divide(1, np.sqrt(2 * np.pi) * s * x) * np.exp(-1 * np.divide(np.square(np.log(x) - m), 2 * s ** 2))


# see print_attributes()
#
# values: the data to be analyzed
def analyze_data(values):
    mu, s, e, v = stats.estimate_lognorm_data_values(values)
    print_attributes(values, mu, s, e, v)


# Trys to read data in path (.petsc or csv data) and returns them in a list.
#
# path:   the path to the data
# l:      the layer for the .petsc file
# return: the data in a list
def get_data(path, l, n):
    v = []
    if ".petsc" in path:
        v = generate_value_array(path, l, n)
    elif ".csv" in path:
        v = values_from_csv(path)[0]
    else:
        mpg.print_error("This file format is not supported!")
        exit(0)
    return v


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='analyze data of metos3d (for data in .petsc files use %i% in their '
                                                 'path as a placeholder for the index (0, to n))')
    parser.add_argument('-a', '--analyze', metavar='path', help='analyze the given data')
    parser.add_argument('-hg', '--histogram', metavar='path', help='generate a histogram of the given data')
    parser.add_argument('-b', '--bins', type=int, help='the number of bins for generating the histogram')
    parser.add_argument('-l', '--layer', type=int, help='the layer of the .petsc data that should be analyzed')
    parser.add_argument('-sp', '--scatter_plot', metavar='path', nargs=2, help='generate a scatter plot '
                                                                               'of the given data')
    parser.add_argument('-r', '--regression', action='store_true', help='generate a regression plot in '
                                                                        'the scatter plot')
    parser.add_argument('-sc', '--second_color', help='the secondary color used in the diagram e.g. for the '
                                                      'regression line or the density function')
    parser.add_argument('-pl', '--plot_lognormal', metavar='path', help='plot the given data as lognormal')
    parser.add_argument('-pr', '--plot_range', type=float, nargs=2, help='the range in which the plot get plotted')
    parser.add_argument('-nov', '--number_of_values', type=int, help='the number of values plotted for the plot '
                                                                     'functions (default 10000)')
    parser.add_argument('-tr', '--x_axis_text_rotation', type=int, help='the rotation of the x axis text ('
                                                                        'default 0)')
    parser.add_argument('-n', '--number', type=int, help='the number of data files you want to read in case you want '
                                                         'to read .petsc files')
    parser.add_argument('-c', '--color', help='the color of the data in the diagrams (default black)')
    parser.add_argument('-o', '--output', metavar='path', help='the path of the file output')
    parser.add_argument('-sl', '--show_l', action='store_true', help='show the General Public License')
    parser.add_argument('-t', '--title', help='the title of the diagrams')
    parser.add_argument('--x_axis', '-xa', help='the title of one or two x-axis of the diagrams')
    parser.add_argument('--y_axis', '-ya', nargs='+', help='the title of the y-axis of the diagrams')
    parser.add_argument('-rt', '--rectangle', type=int, nargs=4, help='just analyze the given rectangle of the data')
    parser.add_argument('-hp', '--histogram_plot', action='store_true', help='generats a approximated plot of an '
                                                                             'lognormal distribution over the '
                                                                             'histogram')

    args = parser.parse_args()

    color = args.color
    if color is None:
        color = "black"

    x_axis = args.x_axis
    if x_axis is not None:
        x_axis = ""

    y_a = args.y_axis
    y_axis = ""
    y_axis2 = ""
    if y_a is not None:
        y_axis = y_a[0]
        y_axis2 = y_a[1]

    title = args.title
    if title is None:
        title = ""

    output = args.output
    if output is None:
        output = "diagram.png"

    layer = args.layer
    if layer is None:
        layer = 0

    num = args.number
    if num is None:
        num = 100

    bins = args.bins
    if bins is None:
        bins = 40

    rotation = args.x_axis_text_rotation
    if rotation is None:
        rotation = 0

    second_color = args.second_color
    if second_color is None:
        second_color = "red"

    show_l = args.show_l
    show_l = False
    show_l = args.show_l
    if show_l:
        mpg.print_license()

    rectangle = args.rectangle
    is_rectangle = rectangle is not None

    analyze = args.analyze
    if analyze is not None:
        values = get_data(analyze, 0, 100)
        analyze_data(values)

    histogram_plot = args.histogram_plot

    histogram = args.histogram
    if histogram is not None:
        values = get_data(histogram, layer, num)
        generate_histogram(values, output, bins, title, color, rotation, histogram_plot, second_color, x_axis, y_axis2,
                           y_axis)

    scatter_plot = args.scatter_plot
    regression = args.regression
    if scatter_plot is not None:
        values1 = get_data(scatter_plot[0], layer, num)
        values2 = get_data(scatter_plot[1], layer, num)
        generate_scatter_plot(values1, values2, output, title, x_axis, y_axis, regression, color, rotation,
                              second_color)

    number_of_values = args.number_of_values
    if number_of_values is None:
        number_of_values = 10000

    plot_range = args.plot_range

    plot_lognormal = args.plot_lognormal
    if plot_lognormal is not None:
        values = get_data(plot_lognormal, layer, num)
        min = stats.get_smallest_number(values)
        max = stats.get_largest_number(values)
        if plot_range is not None:
            min = plot_range[0]
            max = plot_range[1]
        mu, s, e, v = stats.estimate_lognorm_data_values(values)

        plot_lognorm(mu, s, min, max, number_of_values, output, title, x_axis, y_axis, color, rotation)
