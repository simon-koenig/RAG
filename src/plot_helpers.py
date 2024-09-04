import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(data, num_bins=10, file_path="histogram.png"):
    ##
    ## Create a histogram plot and safe it to a file
    ##
    plt.hist(data, bins=num_bins, edgecolor="black", range=(0, 1))

    # Calculate the average
    mean = np.mean(data)

    # Add a field for the mean

    # Add text to the plot for the average
    plt.text(
        0.05,  # Change the x-coordinate to 0.05 for top left
        0.95,
        f"Average: {mean:.2f}",
        transform=plt.gca().transAxes,  # Use Axes coordinates (0,0 is bottom-left, 1,1 is top-right)
        fontsize=12,
        color="black",
        verticalalignment="top",
        horizontalalignment="left",  # Change the horizontal alignment to "left"
    )

    # Add titles and labels
    plt.title("Semantic Similarity Correctness with Rerank")
    plt.xlabel("Correctness")
    plt.ylabel("Frequency")

    # Show the plotif
    plt.savefig(file_path)
