#!/usr/bin/env python3

import sys

import numpy as np
import pyopencl as cl
from PIL import Image


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

    # Create the one-dimensional domain.
    domain_np = np.zeros(width * height, dtype=np.complex64)
    for i in range(width):
        for j in range(height):
            domain_np[i * width + j] = complex(imaginary_limits[0] + i * (imaginary_limits[1] - real_limits[0]) / width,
                                               imaginary_limits[0] + j * (imaginary_limits[1] - imaginary_limits[0]) / width)

    # Create the one-dimensional codomain.
    codomain_np = np.zeros(width * height, dtype=np.uint8)

    # Will either prompt for a context or just go to a default depending on the value of $PYOPENCL_CTX.
    context = cl.create_some_context()

    # Create the command queue.
    queue = cl.CommandQueue(context)

    # Create both domain and codomain buffers.
    domain_gpu = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=domain_np)
    codomain_gpu = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, codomain_np.nbytes)

    # Read the OpenCL C source.
    with open("julia.cl") as program_source_file:
        program_source = program_source_file.read()

    # Compile the source.
    program = cl.Program(context, program_source).build()

    # Run the program.
    program.julia(queue, domain_np.shape, None, domain_gpu, np.complex64(complex(c_real, c_imaginary)),
                  np.uint8(max_iterations), codomain_gpu)

    # Read the results from the device.
    cl.enqueue_copy(queue, codomain_np, codomain_gpu)

    # Reshape the codomain into a 2D array.
    codomain = np.reshape(codomain_np, (width, height)).astype(np.uint8)

    # Save the results.
    image = Image.fromarray(np.transpose(codomain.astype(np.uint8))) # x-axis not in row-major order.
    image.save("Julia.tiff")

    # Exit successfully.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))