""" This module applies majority voting to the predictions in the folder 'majority'. Make sure all the predictions are
in the folder before running the script. This module also allows to specify weights for each predictions file."""

import os
import sys
import getopt
from CONSTANTS import *
import pandas as pd
import random
import csv


def parsing():
    # Remove 1st argument from the list of command line arguments
    arguments = sys.argv[1:]

    # Options
    options = "hw:f:"

    # Long options
    long_options = ["help", "weights=", "filename="]

    # Prepare flags
    flags = {"filename": "predictions" + str(random.randint(0, 100000)) + ".csv"}

    # Parsing argument
    arguments, values = getopt.getopt(arguments, options, long_options)

    if len(arguments) > 0 and arguments[0][0] in ("-h", "--help"):
        print(f'This script performs majority voting on the predictions in the folder majority.\n'
              + '-w or --weights: The weights to apply to the prediction files (default=eqaul weigthts). This argument '
              + 'should be a comma-separated list of weights, with no spaces. If the weights don\'t sum up to 1,'
              + ' normalization is applied. The weights are applied to the files in alphabetical order.\n'
              + '-f or --filename: name of the file for the model (valid name with no extension).')
        sys.exit()

    # checking each argument
    for arg, val in arguments:
        if arg in ("-w", "--weights"):
            weights = [float(num) for num in val.split(",")]
            if sum(weights) != 1.:
                weights = [weight / sum(weights) for weight in weights]
            flags["weights"] = weights

        elif arg in ("-f", "--filename"):
            flags["filename"] = '{}.csv'.format(val)

    return flags


if __name__ == "__main__":
    # Parse command line arguments
    flags = parsing()
    # Loop over files in majority directory and load content into dataframes
    if not os.path.exists(PATH_MAJORITY):
        # If it doesn't exist, create the directory
        raise Exception("Directory majority in not present. Create directory and place predictions files in the "
                        "directory.")
    else:
        # Count number of csv files to produce weights
        all_files = os.listdir(PATH_MAJORITY)

        # Filter the list to only include files with the .csv extension
        num_files = len([file for file in all_files if file.endswith('.csv')])
        if num_files == 0:
            raise Exception("No file is in the directory.")

        if "weights" not in flags.keys():
            # Produce weights
            flags["weights"] = [1 / num_files for _ in range(num_files)]

        dataframes = []
        len_df = -1
        filenames = sorted(os.listdir(PATH_MAJORITY))
        for filename in filenames:
            file_path = os.path.join(PATH_MAJORITY, filename)
            # Check if the current item is a csv file
            if os.path.isfile(file_path) and ".csv" in file_path:
                df = pd.read_csv(file_path, dtype={"Id": int, "Prediction": int})
                if len_df == -1 or df.shape[0] == len_df:
                    df.set_index('Id', inplace=True)
                    dataframes.append(df)
                    len_df = df.shape[0]
                else:
                    raise Exception("The files do not have the same number of predictions.")

        if len(dataframes) == 0:
            raise Exception("The directory is empty. Place at least one file in it.")

        # Compute majority vote
        majority_votes = []
        for ind in dataframes[0].index[1:]:
            weights_0 = 0
            weights_1 = 0
            for i in range(len(dataframes)):
                if dataframes[i].loc[ind, "Prediction"] == -1:
                    weights_0 += flags["weights"][i]
                else:
                    weights_1 += flags["weights"][i]

            # Select value with higher weight
            if weights_0 > weights_1:
                majority_votes.append((ind, -1))
            elif weights_1 > weights_0:
                majority_votes.append((ind, 1))
            else:
                # Select value at random
                majority_votes.append((ind, random.choice([-1, 1])))

        # Save results to csv file
        with open(PATH_PREDICTIONS + "/" + flags["filename"], "w", newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Id', 'Prediction'])
            csv_writer.writerows(majority_votes)
