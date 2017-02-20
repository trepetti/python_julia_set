#!/usr/bin/env python3

import sys

import numpy as np
from PIL import Image


def julia(z, c, max_iterations):
    """
    Julia set function to be vectorized. Inspired by Mayavi documentation.
    """

    # Calculate the divergence over the requested number of iterations.
    n = 0
    while n < max_iterations and np.abs(z) < 2.0:
        z = z * z + c
        n += 1

    # Should be a bitmap value ranged 0 to max_iterations.
    return n


vectorized_julia = np.vectorize(julia)


def main(argv):
    """
    Entry point.
    """

    # Check usage.
    if len(argv) != 3:
        print("Error: expected two arguments of the form: [Re(c)] [Im(c)].", file=sys.stderr)
        return 1

    # Make sure the arguments all make sense.
    try:
        c_real = float(argv[1])
        c_imaginary = float(argv[2])
    except ValueError as error:
        print("Error: expected two arguments of the form: [Re(c)] [Im(c)].\n"
              + "At least one of the arguments given was of the wrong type.", file=sys.stderr)
        return 1

    # Set static test parameters.
    width = 1024
    height = 1024
    real_limits = (-1.5, 1.5)
    imaginary_limits = (-1.5, 1.5)
    max_iterations = 255

    # Create the domain.
    domain = np.zeros((width, height), dtype=np.complex64)
    for i in range(width):
        for j in range(height):
            domain[i][j] = complex(real_limits[0] + i * (real_limits[1] - real_limits[0]) / width,
                                   imaginary_limits[0] + j * (imaginary_limits[1] - imaginary_limits[0]) / width)

    # Calculate the set.
    codomain = vectorized_julia(domain, np.complex64(complex(c_real, c_imaginary)), max_iterations)

    # Save the results.
    image = Image.fromarray(np.transpose(codomain.astype(np.uint8))) # x-axis not in row major order.
    image.save("Julia.tiff")

    # Exit successfully.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))