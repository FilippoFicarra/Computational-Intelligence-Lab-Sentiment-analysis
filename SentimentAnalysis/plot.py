import math
import random

import matplotlib.pyplot as plt
import os

import torch
import numpy as np

from CONSTANTS import *


def plot_accuracy_or_loss(values, title, label, y_label, filename):
    # Plot
    trn_y_values = torch.tensor(values)[:, 1]

    # Get ticks for x-axis
    i = 0
    ticks = []
    for row in values:
        if row[0] == 0 and i != 0:
            ticks.append(i - 1)
        i += 1

    ticks.append(trn_y_values.size(0) - 1)
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(0, i)), trn_y_values, label=label, color='darkorange' if "val" in filename else "blue")
    plt.xticks(ticks, list(range(1, len(ticks) + 1)))

    if "losses" in filename:
        plt.yticks(np.arange(0.05, 0.7 + 0.05, 0.05))
    elif "accuracies" in filename:
        # Get the minimum value from the tensor
        min_value = torch.min(trn_y_values)
        # Round the minimum value to the closest multiple of 5, which is lower than the value
        rounded_min_value = 5 * math.floor(min_value.item() / 5)
        plt.yticks(range(rounded_min_value, 105, 5))

    plt.title(title)
    plt.xlabel(X_LABEL)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig("figures/" + filename + ".png")


if __name__ == "__main__":
    all_files = os.listdir(PATH_LOSSES_AND_ACCURACIES)

    # Filter files ending with .pt extension
    pt_files = [file for file in all_files if file.endswith(".pt")]

    num = random.randint(1, 100000)

    for filename in pt_files:
        file_path = os.path.join(PATH_LOSSES_AND_ACCURACIES, filename)

        if "trn" in filename:
            if "accuracies" in filename:
                plot_accuracy_or_loss(torch.load(file_path), TITLE_TRAINING_ACCURACY, LABEL_TRAINING_ACCURACY,
                                      Y_LABEL_ACCURACY, filename.replace(".pt", "") + "-" + str(num))
            elif "losses" in filename:
                plot_accuracy_or_loss(torch.load(file_path), TITLE_TRAINING_LOSS, LABEL_TRAINING_LOSS, Y_LABEL_LOSS,
                                      filename.replace(".pt", "") + "-" + str(num))
        elif "val" in filename:
            if "accuracies" in filename:
                plot_accuracy_or_loss(torch.load(file_path), TITLE_VAL_ACCURACY, LABEL_VAL_ACCURACY, Y_LABEL_ACCURACY,
                                      filename.replace(".pt", "") + "-" + str(num))
            elif "losses" in filename:
                plot_accuracy_or_loss(torch.load(file_path), TITLE_VAL_LOSS, LABEL_VAL_LOSS, Y_LABEL_LOSS,
                                      filename.replace(".pt", "") + "-" + str(num))
