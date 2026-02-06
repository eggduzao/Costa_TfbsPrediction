"""
This script will generate input-out plots for all of the activation
functions. These are for use in the documentation, and potentially in
online tutorials.
"""

from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import smith


matplotlib.use("Agg")


# Create a directory for the images, if it doesn't exist
ACTIVATION_IMAGE_PATH = Path(__file__).parent / "activation_images"

if not ACTIVATION_IMAGE_PATH.exists():
    ACTIVATION_IMAGE_PATH.mkdir()

# In a refactor, these ought to go into their own module or entry
# points so we can generate this list programmatically
functions = [
    smith.nn.ELU(),
    smith.nn.Hardshrink(),
    smith.nn.Hardtanh(),
    smith.nn.Hardsigmoid(),
    smith.nn.Hardswish(),
    smith.nn.LeakyReLU(negative_slope=0.1),
    smith.nn.LogSigmoid(),
    smith.nn.PReLU(),
    smith.nn.ReLU(),
    smith.nn.ReLU6(),
    smith.nn.RReLU(),
    smith.nn.SELU(),
    smith.nn.SiLU(),
    smith.nn.Mish(),
    smith.nn.CELU(),
    smith.nn.GELU(),
    smith.nn.Sigmoid(),
    smith.nn.Softplus(),
    smith.nn.Softshrink(),
    smith.nn.Softsign(),
    smith.nn.Tanh(),
    smith.nn.Tanhshrink(),
    smith.nn.Threshold(0, 0.5),
    # Note: GLU is not included because it requires splitting the input tensor
    # into two halves (a, b) and computing a * sigmoid(b). A simple 1D input-output
    # plot doesn't meaningfully represent this behavior.
]


def plot_function(function, **args):
    """
    Plot a function on the current plot. The additional arguments may
    be used to specify color, alpha, etc.
    """
    xrange = smith.arange(-7.0, 7.0, 0.01)  # We need to go beyond 6 for ReLU6
    x = xrange.numpy()
    y = function(xrange).detach().numpy()
    plt.plot(x, y, **args)


# Step through all the functions
for function in functions:
    function_name = function._get_name()
    plot_path = ACTIVATION_IMAGE_PATH / f"{function_name}.png"
    if not plot_path.exists():
        # Start a new plot
        plt.clf()
        plt.grid(color="k", alpha=0.2, linestyle="--")

        # Plot the current function
        plot_function(function)

        plt.title(function)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])

        # And save it
        plt.savefig(plot_path)
        print(f"Saved activation image for {function_name} at {plot_path}")
