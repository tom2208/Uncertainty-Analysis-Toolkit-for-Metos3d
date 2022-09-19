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

import sys
import numpy as np
import yaml
from colorama import Fore
import hist4cmd as hist
import statistics as stats
import csv

# notice
notice = "Metos3D-Parameter-Generator Copyright (C) 2022 Tom L. Hauschild.\nThis program comes with ABSOLUTELY NO " \
         "WARRANTY. This is free software,\nand you are welcome to redistribute it under certain conditions.\n" \
         "Type `--show_l' for details."

# error messages
option_read_error = "Couldn't read template option file: '"
indicator_key_not_found = "Error: Indicator key not found, continuing without setting it"

# distribution variable
variable = "%Di%"
config_dir = "resources/config.yaml"

# arguments passed to the program
debug = False
quiet = False
print_array = False
display_histogram = False
histogram_height = 15
histogram_width = 3
histogram_spacing = 1
histogram_buckets = 15

# dictionary for option file indicators -> their replacements (init: standard values)
indicator_replacements = {
    # debug
    "%Metos3dDebugLevel%": "3",
    "%Metos3DGeometryType%": "Profile",
    "%Metos3DProfileInputDirectory%": "data/TMM/2.8/Geometry/",
    "%Metos3DProfileMaskFile%": "landSeaMask.petsc",
    "%Metos3DProfileVolumeFile%": "volumes.petsc",

    # bgc tracer
    "%Metos3DTracerCount%": "1",
    "%Metos3DTracerName%": "N",
    "%Metos3DTracerInitValue%": "2.17e+0",
    "%Metos3DTracerOutputDirectory%": "work/",
    "%Metos3DTracerOutputFile%": "N%i%.petsc",

    # diagnostic variables
    "%Metos3DDiagnosticCount%": "0",

    # bgc parameter
    "%Metos3DParameterCount%": "5",
    "%Metos3DParameterValue%": "0.02,2.0,0.5,30.0,0.858",

    # bgc boudary conditions
    "%Metos3DBoundaryConditionCount%": "2",
    "%Metos3DBoundaryConditionInputDirectory%": "data/TMM/2.8/Forcing/BoundaryCondition/",
    "%Metos3DBoundaryConditionName%": "Latitude,IceCover",

    # latitude
    # ice cover
    "%Metos3DLatitudeCount%": "1",
    "%Metos3DLatitudeFileFormat%": "latitude.petsc",
    "%Metos3DIceCoverCount%": "12",
    "%Metos3DIceCoverFileFormat%": "fice_$02d.petsc",

    # bgc domain conditions
    "%Metos3DDomainConditionCount%": "2",
    "%Metos3DDomainConditionInputDirectory%": "data/TMM/2.8/Forcing/DomainCondition/",
    "%Metos3DDomainConditionName%": "LayerDepth,LayerHeight",

    # layer depth
    "%Metos3DLayerDepthCount%": "1",
    "%Metos3DLayerDepthFileFormat%": "z.petsc",

    # layer height
    "%Metos3DLayerHeightCount%": "1",
    "%Metos3DLayerHeightFileFormat%": "dz.petsc",

    # transport
    "%Metos3DTransportType%": "Matrix",
    "%Metos3DMatrixInputDirectory%": "data/TMM/2.8/Transport/Matrix5_4/1dt/",
    "%Metos3DMatrixCount%": "2",
    "%Metos3DMatrixExplicitFileFormat%": "Ae_$02d.petsc",
    "%Metos3DMatrixImplicitFileFormat%": "Ai_$02d.petsc",

    # time stepping
    "%Metos3DTimeStepStart%": "0.0",
    "%Metos3DTimeStepCount%": "1",
    "%Metos3DTimeStep%": "0.0003472222222222",

    # solver
    "%Metos3DSolverType%": "Spinup",
    "%Metos3DSpinupCount%": "1",
}


# Writes the option files to the directory "filepath" replacing the variables with its indented values.
#
# filepath:              the directory to write the option files to
# option_file_content:   the content of the option file
# variable_replacements: a list of values to replace the variables with, beginning with the first value being associated
#                        with the first distribution "D0" and so on.
# index:                 a index list for values of the variable_replacements
def write_option_file(filepath, option_file_content, variable_replacements, index):
    try:
        with open(filepath, "w") as file_stream:

            for i in range(len(variable_replacements)):
                option_file_content = option_file_content.replace(variable.replace("i", str(i)),
                                                                  str(variable_replacements[i][index[i]]))

            file_stream.write(option_file_content)
            file_stream.close()
    except FileExistsError:
        print_error(option_read_error + filepath + "'")
        sys.exit(0)


# Writes the "content" to the file "filepath".
#
# filepath: the file to write to
# content:  the content to write to the file
def write_txt_file(filepath, content):
    with open(filepath, "w") as file_stream:
        file_stream.write(content)
        file_stream.close()


# Writes the data to a csv file.
#
# filepath: the file to write to
# data:     the data to write to the file
def write_csv_file(filepath, data):
    with open(filepath, "w") as file_stream:
        writer = csv.writer(file_stream)
        writer.writerows([data])
        file_stream.close()


# Reads and returns the option file.

# returns: the option file content
def read_option_file(path):
    try:
        with open(path) as file_stream:
            lines = file_stream.readlines()
            file_stream.close()
            return lines
    except FileNotFoundError:
        print_error(option_read_error + path + "'")
        sys.exit(0)


# Reads and returns the config file.

# returns: the config file content
def read_yaml_file():
    with open(config_dir) as fileStream:
        try:
            loaded = yaml.safe_load(fileStream)
        except yaml.YAMLError as exception:
            print(exception)
            sys.exit(0)

    if loaded:
        return loaded


# Replaces all indicators in "lines" and replaces '%i%' with the "index" of the option file.
#
# lines:   the lines to replace the indicators in
# index:   the index of the option file
# returns: the lines with the replaced indicators
def replace_indicators(lines, index):
    string = ""
    for line in lines:
        for indicator in indicator_replacements:
            replacement = indicator_replacements[indicator]
            if indicator == "%Metos3DTracerOutputFile%":
                replacement = replacement.replace("%i%", str(index))
            line = line.replace(indicator, str(replacement))
        string += line
    return string


# Uses yaml data to replace the default values in the dictionary "indicator_replacements".
#
# yaml_data: the yaml data to use to replace the default values
def set_data_from_yaml(yamlData):
    for key in yamlData["model"]:
        alt_key = "%" + key + "%"
        if alt_key in indicator_replacements:
            indicator_replacements[alt_key] = yamlData["model"][key]
        else:
            indicator_replacements.update({alt_key : yamlData["model"][key]})


# Print an error message with the content "message".
#
# message: the message to print
def print_error(message):
    if not quiet:
        print(Fore.RED + "[ERROR] " + message + Fore.RESET)


# Print a warning message with the content "message".
#
# message: the message to print
def print_warning(message):
    if not quiet:
        print(Fore.YELLOW + "[WARNING] " + message + Fore.RESET)


# Print a debug message with the content "message".
#
# message: the message to print
def print_debug(message):
    if debug:
        print(Fore.BLUE + "[DEBUG] " + Fore.RESET + message + Fore.RESET)


# Print a success message with the content "message".
#
# message: the message to print
def print_success(message):
    if not quiet:
        print(Fore.GREEN + "[SUCCESS] " + message + Fore.RESET)


# Print an info message with the content "message".
def print_info(message):
    if not quiet:
        print(Fore.CYAN + "[INFO] " + Fore.RESET + message)


# Prints a separator message using the character "-".
def print_seperator():
    if not quiet:
        print("------------------------------------------------------------------------------")


# Prints a separator message using the character "=".
def print_double_seperator():
    if not quiet:
        print("==============================================================================")


# Generates a string with mpirun commands using yaml data.
#
# yamlData: the yaml data to use to generate the mpirun commands
# names:    the names of the option to be included in the mpirun commands
# returns:  the string with the mpirun commands
def generate_mpirun(yaml_data, names):
    program_path = yaml_data["mpirun"]["program_path"]
    optionfiles_path = yaml_data["mpirun"]["optionfiles_path"]
    options = yaml_data["mpirun"]["options"]

    arguments = ""
    for name in names:
        arguments += "mpirun " + options + " " + program_path + " " + optionfiles_path + str(name) + "\n"

    return arguments[0:len(arguments) - 1]


# The main function. It generates all files and data depending on the configuration by the config file and arguments
# passed to the program.
def generate_option_files():
    yaml_data = read_yaml_file()
    number_of_distributions = yaml_data["distributions"]["number"]
    file_name = yaml_data["file_name"]
    output_directory = yaml_data["output_directory"]

    value_array = [0 for i in range(number_of_distributions)]

    for i in range(number_of_distributions):
        value_array[i] = generate_random_parameter(i, yaml_data)
        if display_histogram:
            print_info("Histogram of D" + str(i) + ": ")
            print()
            hist.display_histogram(value_array[i], histogram_buckets, histogram_height, histogram_width,
                                   histogram_spacing)
            print_double_seperator()
        if yaml_data["distributions"]["D" + str(i)]["save_in_csv"]:
            print_debug("Saving D" + str(i) + " in csv file: ")
            write_csv_file(output_directory + "D" + str(i) + ".csv", value_array[i])
            print_info("Saved D" + str(i) + " in csv file: " + output_directory + "D" + str(i) + ".csv")

    set_data_from_yaml(yaml_data)
    option_file_path = yaml_data["option_file_path"]
    content = read_option_file(option_file_path)

    sample_size = len(value_array[0])
    if number_of_distributions >= 2:
        sample_size *= len(value_array[1])
    elif number_of_distributions >= 3:
        sample_size *= len(value_array[2])

    option_file_names = [0 for i in range(sample_size)]

    k = 0
    while k == 0 or (number_of_distributions >= 3 and k < len(value_array[2])):
        j = 0
        while j == 0 or (number_of_distributions >= 2 and j < len(value_array[1])):
            for i in range(len(value_array[0])):
                option_file_name = file_name + str(i) + "-" + str(j) + "-" + str(k) + ".txt"
                len1 = 0
                len2 = 0
                if number_of_distributions >= 2:
                    len1 = len(value_array[1])
                if number_of_distributions >= 3:
                    len2 = len(value_array[2])

                index = (k * len2)+(j * len1)+i
                option_file_names[index] = option_file_name
                write_option_file(output_directory + option_file_name, replace_indicators(content, i), value_array,
                                  [i, j, k])
            j += 1
        k += 1

    # generate command arguments
    if yaml_data["mpirun"]["generate"]:
        print_debug("Generating mpirun commands... ")
        write_txt_file(output_directory + "mpirun.txt", generate_mpirun(yaml_data, option_file_names))
        print_info("Generated mpirun commands: " + output_directory + "mpirun.txt")

    print_success("Option files generated.")


# Generates random values from the distributions configured by the yaml_data. The function then prints out attributes
# of the distributions of the generated values (i.a. expected values, variances and the deltas / differences between
# generated and entered parameters)
#
# distribution_index: the index of the distribution to generate the values for
# yaml_data:          the yaml data to use to generate the random values
# returns:            the array with the generated values
def generate_random_parameter(distribution_index, yaml_data):
    print_info("Generating random parameter values for distribution D" + str(distribution_index) + "...")

    distribution_type = yaml_data["distributions"]["D" + str(distribution_index)]["type"]
    lower_bound = yaml_data["distributions"]["D" + str(distribution_index)]["lower_bound"]
    upper_bound = yaml_data["distributions"]["D" + str(distribution_index)]["upper_bound"]
    sample_size = yaml_data["distributions"]["D" + str(distribution_index)]["sample_size"]
    tries = yaml_data["distributions"]["D" + str(distribution_index)]["tries"]
    value_on_fail = yaml_data["distributions"]["D" + str(distribution_index)]["value_on_fail"]

    parameter_array = [0 for i in range(sample_size)]
    number_of_failed_generations = 0

    for i in range(sample_size):
        new_parameter = 0
        for j in range(tries):
            if distribution_type == "lognormal":
                new_parameter = np.random.lognormal(yaml_data["distributions"]["D" + str(distribution_index)]["mu"],
                                                    yaml_data["distributions"]["D" + str(distribution_index)][
                                                        "sigma"])
            elif distribution_type == "normal":
                new_parameter = np.random.normal(yaml_data["distributions"]["D" + str(distribution_index)]["mean"],
                                                 yaml_data["distributions"]["D" + str(distribution_index)]["variance"])
            elif distribution_type == "geometric":
                new_parameter = np.random.geometric(
                    yaml_data["distributions"]["D" + str(distribution_index)]["probability"])
            elif distribution_type == "poisson":
                new_parameter = np.random.poisson(yaml_data["distributions"]["D" + str(distribution_index)]["lambda"])
            elif distribution_type == "exponential":
                new_parameter = np.random.exponential(
                    yaml_data["distributions"]["D" + str(distribution_index)]["lambda"])
            elif distribution_type == "uniform":
                new_parameter = np.random.uniform(yaml_data["distributions"]["D" + str(distribution_index)]["lower"],
                                                  yaml_data["distributions"]["D" + str(distribution_index)]["upper"])
            else:
                print_error("Distribution type not found: " + distribution_type)
                exit(1)

            if lower_bound <= new_parameter <= upper_bound:
                parameter_array[i] = new_parameter
                break

        if not (lower_bound <= new_parameter <= upper_bound):
            number_of_failed_generations += 1
            parameter_array[i] = value_on_fail
            print_debug("Failed to generate value for parameter " + str(i) + ": " + str(new_parameter))
            if number_of_failed_generations == sample_size:
                print_error("Failed to generate all parameter values. Exiting.")
                exit(1)

    if print_array:
        print_info("Generated values for distribution D" + str(distribution_index) + ": " + str(parameter_array))

    if not (number_of_failed_generations == 0):
        print_warning("Values for distribution D" + str(distribution_index) + " generated. Failed to generate " +
                      str(number_of_failed_generations) + " parameters after " + str(tries) +
                      " tries. They will be set to " + str(value_on_fail) + ".")
    else:
        print_success("Random parameter values for distribution D" + str(distribution_index) + " successfully "
                                                                                               "generated.")
    if distribution_type == "lognormal":
        mu = yaml_data["distributions"]["D" + str(distribution_index)]["mu"]
        sigma = yaml_data["distributions"]["D" + str(distribution_index)]["sigma"]
        anticipated_values = stats.lognorm_values(mu, sigma)
        estimated_values = stats.estimate_lognorm_data_values(parameter_array)

        print_double_seperator()
        print_info("Anticipated values for distribution D" + str(distribution_index) + ": ")
        print_double_seperator()
        print_info("used mu:\t\t\t" + str(mu))
        print_info("used sigma:\t\t" + str(sigma))
        print_info("expected value:\t\t" + str(anticipated_values[0]))
        print_info("expected variance:\t" + str(anticipated_values[1]))
        print_seperator()
        print_info("estimated mu:\t\t" + str(estimated_values[0]))
        print_info("estimated sigma:\t\t" + str(estimated_values[1]))
        print_info("estimated expected value:" + str(estimated_values[2]))
        print_info("estimated variance:\t" + str(estimated_values[3]))
        print_seperator()
        print_info("mu delta:\t\t" + str(np.absolute(mu - estimated_values[0])))
        print_info("sigma delta:\t\t" + str(np.absolute(sigma - estimated_values[1])))
        print_info("expected value delta:\t" + str(np.absolute(anticipated_values[0] - estimated_values[2])))
        print_info("variance delta:\t\t" + str(np.absolute(anticipated_values[1] - estimated_values[3])))
        print_double_seperator()

    return parameter_array


# Prints out the license
def print_license():
    print_double_seperator()
    with open('LICENSE.txt') as f:
        lines = f.readlines()
    print("".join(str(x) for x in lines))


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='create option files for metos3d')
    parser.add_argument('--config', metavar='path', required=True, help='set the path to the config.yaml')
    parser.add_argument('-d', '--debug', action='store_true', help='enable debug mode for more information output')
    parser.add_argument('-pa', '--print-array', action='store_true', help='print all generated parameters after '
                                                                          'generating them')
    parser.add_argument('-q', '--quiet', action='store_true', help='disable all outputs')
    parser.add_argument('-dh', '--display_histogram', action='store_true', help='display a histogram of the generated '
                                                                                'values')
    parser.add_argument('-hh', '--histogram_height', type=int, help='set the height of the histogram in characters')
    parser.add_argument('-hb', '--histogram_buckets', type=int, help='set the number of buckets used in the '
                                                                     'histogram, the histogram will be at least as '
                                                                     'wide as the number of buckets')
    parser.add_argument('-hw', '--histogram_width', type=int, help='set the number of characters a single bar of the '
                                                                   'histogram should be wide')
    parser.add_argument('-hs', '--histogram_spacing', type=int, help='set the number of spaces between the bars of '
                                                                     'the histogram')
    parser.add_argument('-sl', '--show_l', action='store_true', help='show the General Public License')

    args = parser.parse_args()
    config_dir = args.config
    debug = args.debug
    quiet = args.quiet
    print_array = args.print_array
    display_histogram = args.display_histogram

    if args.histogram_height is not None:
        histogram_height = args.histogram_height
    if args.histogram_buckets is not None:
        histogram_buckets = args.histogram_buckets
    if args.histogram_width is not None:
        histogram_width = args.histogram_width
    if args.histogram_spacing is not None:
        histogram_spacing = args.histogram_spacing

    show_l = False
    show_l = args.show_l
    if show_l:
        print_license()

    print_double_seperator()
    print(notice)
    print_double_seperator()
    generate_option_files()
